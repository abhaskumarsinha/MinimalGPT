import numpy as np
import pandas as pd
import tensorflow as tf
import math
from tqdm import tqdm

def scaled_dot_product_attention(q, k, v):
    # calculate the dot product of query and key
    dot_product = tf.matmul(q, k, transpose_b=True)
    
    
    # scale the dot product
    scaled_dot_product = dot_product / tf.math.sqrt(tf.cast(tf.shape(k)[-1], dtype=tf.float32))
    
    # apply softmax activation to obtain attention weights
    attention_weights = tf.nn.softmax(scaled_dot_product, axis=-1)
    
    # compute the weighted sum of the value vectors with attention weights
    output = tf.matmul(attention_weights, v)
    
    return output


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, ix, ox):
        super().__init__()
        self.ix = ix
        self.ox = ox
        
    
    def build(self, input_shapes):
        self.w1 = self.add_weight(shape=(self.ix, self.ox))
        self.b1 = self.add_weight(shape=(1, self.ox))
    
    def call(self, inputs):
        bz, key = tf.shape(inputs)[0], tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, (-1, self.ix))
        inputs = tf.matmul(inputs, self.w1) + self.b1
        inputs = tf.reshape(inputs, (bz, key, self.ox))
        return inputs
    


class split_heads(tf.keras.layers.Layer):
    def __init__(self, num_heads = 10):
        super().__init__()
        self.num_heads = num_heads
    
    def call(self, inputs):
        bz, key = tf.shape(inputs)[0], tf.shape(inputs)[1]
        
        inputs = tf.reshape(inputs, (bz, key, self.num_heads, -1))
        inputs = tf.transpose(inputs, (0, 2, 1, 3))
        
        return inputs

    
class merge_heads(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs):
        bz, key = tf.shape(inputs)[0], tf.shape(inputs)[2]
        
        inputs = tf.transpose(inputs, (0, 2, 1, 3))
        inputs = tf.reshape(inputs, (bz, key, -1))
        return inputs

    

class GPT_Attention(tf.keras.layers.Layer):
    
    def __init__(self, ix, ox, num_heads):
        super().__init__()
        self.ix = ix
        self.ox = ox
        self.num_heads = num_heads
        self.linear1 = LinearLayer(self.ix, self.ox * 3)
        self.split = split_heads(num_heads = self.num_heads)
        self.merge = merge_heads()
        self.linear2 = LinearLayer(self.ox, self.ix)
        
        if self.ox % self.num_heads != 0:
            raise ValueError('The value ox = '+ str(self.ox) +' SHOULD be divisible by number of heads provided')
    
    def call(self, inputs):
        if len(inputs) > 0:
            inputs = inputs[0]
        inputs = self.linear1(inputs)
        k, q, v = tf.split(inputs, 3, axis = -1)
        k = self.split(k)
        q = self.split(q)
        v = self.split(v)
        #k, q, v = tf.split(inputs, 3, axis = -1)
        inputs = scaled_dot_product_attention(k, q, v)
        inputs = self.merge(inputs)
        inputs = self.linear2(inputs)
               
        return inputs



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads = 8, key_dim = 64, key_embedding = 512):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.key_embedding = key_embedding
        self.head_vectors = []
    
    def build(self, input_shape):
        #print(input_shape)
        
        self.W_k = self.add_weight(shape=(self.num_heads, self.key_dim, self.key_embedding), name='key')
        self.W_q = self.add_weight(shape=(self.num_heads, self.key_dim, self.key_embedding), name='query')
        self.W_v = self.add_weight(shape=(self.num_heads, self.key_dim, self.key_embedding), name='value')
        
        self.W_o = self.add_weight(shape=(self.key_dim, self.key_embedding))
        

    def call(self, inputs):
        query, key, value = inputs
        
        self.head_vectors = []
        head_concat = None
        
        for i in range(self.num_heads):
            q = tf.einsum('bij, ij -> bij', query, self.W_q[i])
            k = tf.einsum('bij, ij -> bij', key, self.W_k[i])
            v = tf.einsum('bij, ij -> bij', value, self.W_v[i])
            
            self.head_vectors += [scaled_dot_product_attention(q, k, v)]
            
            
        head_concat = tf.concat(self.head_vectors, -2)
        #print(tf.shape(head_concat))
        output =tf.einsum('bij, kj -> bkj', head_concat, self.W_o)
            
        
        return output
        
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_heads = 8, key_dim = 64, key_embedding = 512, GPT_attention = False):
        super(Decoder, self).__init__()
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.key_embedding = key_embedding
        if GPT_attention:
            self.attention = GPT_Attention(key_embedding, key_embedding, num_heads)
        else:
            self.attention = MultiHeadAttention(num_heads = num_heads, key_dim = key_dim, key_embedding = key_embedding)
        self.normalize1 = tf.keras.layers.LayerNormalization(axis = -2)
        self.normalize2 = tf.keras.layers.LayerNormalization(axis = -2)
        
        
    def build(self, input_shape):
        #print(input_shape)
        
        self.x1 = self.add_weight(shape=(self.key_dim, self.key_embedding), name='vec1')
        self.x2 = self.add_weight(shape=(self.key_dim, self.key_embedding), name='vec2')
        
        self.y1 = self.add_weight(shape=(self.key_dim, self.key_embedding), name='bias1')
        self.y2 = self.add_weight(shape=(self.key_dim, self.key_embedding), name='bias2')
        
    def call(self, inputs):
        
        first_sublayer_output = self.attention((inputs, inputs, inputs))
        first_sublayer_output = self.normalize1(first_sublayer_output + inputs)
        
        first_nn = tf.einsum('bij, ij -> bij', first_sublayer_output, self.x1) + self.y1
        first_nn = tf.keras.activations.relu(first_nn, alpha=0.0, max_value=None, threshold=0.0)
        second_nn = tf.einsum('bij, ij -> bij', first_nn, self.x2) + self.y2
        
        second_sublayer_output = self.normalize2(second_nn + first_sublayer_output)
        
        
        
        return second_sublayer_output

def positional_function(words, embedding):
    pos = np.zeros((words, embedding))
    
    for i in range(words):
        for j in range(embedding):
            if j%2 == 0:
                pos[i, j] = math.sin(i/pow(10000, 2*j/(512)))
            else:
                pos[i, j] = math.cos(i/pow(10000, 2*j/(512)))
    
    return pos


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, positional_function = positional_function, embedding_size = 512, words = 64):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.words = words
        self.pos_mat = tf.cast(tf.convert_to_tensor(positional_function(self.words, self.embedding_size)), tf.float32)
        
    def build(self, input_sizes):
        print(input_sizes)
        
    def call(self, inputs):
        embed = tf.einsum("bij, ij -> bij", inputs, self.pos_mat)            
        return embed

def generate_output(model, vectorizer, text_size = 70, gpt_input = 64, input_sequence = []):
    
    if input_sequence == []:
        input_sequence = tf.zeros((1, gpt_input)).numpy()
    
    text = tf.zeros((1, text_size)).numpy()
    text[0][: gpt_input] = input_sequence[0][: gpt_input]
    
    GPT = model
    
    
    for i in tqdm(range(gpt_input, text_size)):
        #print("Iteration number:" + str(i))
        output = tf.argmax(GPT(input_sequence), -1).numpy()
        text[0][i - 1] = output
        input_sequence = text[0][i - gpt_input : i].reshape(1, gpt_input)
    
    op = [vectorizer.get_vocabulary()[int(text[0][i])] for i in range(len(text[0]))]
    return ' '.join(op)
import os
import json
import tensorflow as tf
from tqdm import tqdm
from GPT import *
import pickle
import argparse
import sys



def save_module(save_weights, model, vectorizer, save_tokenizer):
    
        # Save the GPT Model
        with open(save_weights, 'wb') as file:
            pickle.dump(model.weights, file)
        
        #Save the Vectorizer Model
        vocabulary = vectorizer.get_vocabulary()

        # Encode the vocabulary as JSON-compatible strings
        encoded_vocabulary = [word.encode('unicode_escape').decode('utf-8') for word in vocabulary]
        encoded_vocabulary = encoded_vocabulary[2:]

        # Save the encoded vocabulary to a JSON file
        with open(save_tokenizer, 'w') as f:
            json.dump(encoded_vocabulary, f)
            print("Vocabulary size saved: " + str(len(encoded_vocabulary)))
    
    
    
    

def read_file(f, vectorizer, chunk_size = 1024, starting_chunk = 0, ending_chunk = 5, gpt_input = 10):
    i = 0
    chunk = []
    
    while True:
        data = f.read(chunk_size)
        
        if not data or i > ending_chunk:
            break
        
        if i >= starting_chunk and i <= ending_chunk:
            file_contents = data.split()
            input_tokens, output_tokens = [], []
            for j in range(len(file_contents) - gpt_input - 1):
                input_tokens += [file_contents[j : j + gpt_input]]
                output_tokens += [file_contents[j + gpt_input]]
               
            
            X = [' '.join(input_tokens[j]) for j in range(len(input_tokens))]
            Y = output_tokens
            
            X = vectorizer(X)
            Y = vectorizer(Y)
            
            output = tf.concat([X, Y], 1)
        
            yield output
        
        i += 1


def get_model(gpt_input, d_model, h, vocab_size, decoder_stacks, GPT_attention):
    input_words = tf.keras.layers.Input((gpt_input))
    embedding = tf.keras.layers.Embedding(vocab_size + 2, d_model)(input_words)
    positional_enc = PositionalEmbedding(words = gpt_input, embedding_size = d_model)(embedding)
    decoder = Decoder(num_heads = h, key_dim = gpt_input, key_embedding = d_model, GPT_attention = GPT_attention)(positional_enc)
    
    for _ in range(decoder_stacks - 1):
        decoder = Decoder(num_heads = h, key_dim = gpt_input, key_embedding = d_model, GPT_attention = GPT_attention)(decoder)
    
    decoder = tf.keras.layers.Flatten()(decoder)
    linear_layer = tf.keras.layers.Dense(vocab_size + 3)(decoder)
    softmax = tf.nn.softmax(linear_layer)
    GPT = tf.keras.Model(inputs = input_words, outputs = softmax)
    
    return GPT


def MinimalGPT(data_path='.', 
               learning_rate=0, 
               output_length=0, 
               epochs = 1, 
               batch_size = 1, 
               gpt_input=10, 
               d_model=128, 
               h=8, 
               decoder_stacks=1, 
               starting_chunk = 0,
               ending_chunk = 5,
               chunk_size = 10,
               token_end=40000,
               vocabulary_start = 0,
               vocabulary_end = 40000,
               save=False, 
               load_tokenizer=None, 
               load_weights=None, 
               save_tokenizer=None,
               save_weights=None,
               optimizer=None,
               inference_only = False,
               return_model_and_vectorizer = False,
               return_model_and_vectorizer_and_output = False,
               GPT_attention = False,
               TPU = False):
    
    if chunk_size:
        chunk_size *= 1024
    
    
    if inference_only == False:
        with open(data_path, 'r', encoding = 'utf-8') as file:
            corpus = file.read()
            #file_contents = corpus.split()[token_start : token_end]
            #print("Total tokens: " + str(len(file_contents)))
            
    
    if load_tokenizer:
            with open(load_tokenizer, 'r') as f:
                encoded_vocabulary = json.load(f)

            # Decode the encoded vocabulary to original strings
            vocabulary = [word.encode('utf-8').decode('unicode_escape') for word in encoded_vocabulary]
            vectorizer = tf.keras.layers.TextVectorization(standardize = None, split = 'whitespace')
            vectorizer.set_vocabulary(vocabulary)
            vocab_size = vectorizer.vocabulary_size()
            
    else:
        vocab = []
        for word in tqdm(corpus.split()[vocabulary_start : vocabulary_end]):
            vocab += [word]
            vocab = list(set(vocab))
        vocab_size = len(vocab)
        vectorizer = tf.keras.layers.TextVectorization(standardize = None, split = 'whitespace', vocabulary = vocab)
        print('New Vectorizer created successfully...')
        print("Vocabulary Size: " + str(vocab_size))   
        del corpus
        
    
    #if inference_only == False:
    #    input_tokens, output_tokens = [], []
    #    for i in tqdm(range(len(file_contents) - gpt_input - 1)):
    #        input_tokens += [file_contents[i : i + gpt_input]]
    #        output_tokens += [file_contents[i + gpt_input]]
               
            
    #    X = [' '.join(input_tokens[i]) for i in tqdm(range(len(input_tokens)))]
    #    Y = output_tokens
    
    #    del corpus
    
    #   X = vectorizer(X)
    #    Y = vectorizer(Y)
    
    if load_weights:
        model = get_model(gpt_input = gpt_input, d_model = d_model, h = h, decoder_stacks = decoder_stacks, vocab_size = vocab_size - 2, GPT_attention = GPT_attention)
        
        with open(load_weights, 'rb') as file:
            W = pickle.load(file)
            model.set_weights(W)
    else:
        model = get_model(gpt_input = gpt_input, d_model = d_model, h = h, decoder_stacks = decoder_stacks, vocab_size = vocab_size, GPT_attention = GPT_attention)
    
    
    print(model.summary())
    
    
    if inference_only == False:
        # Compile the model
        if not optimizer:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy')
        else:
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        
        # Train the model
        if learning_rate > 0:
            
            for epoch in tqdm(range(epochs)):
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    chunk_number = 1
                    for chunk in read_file(f,
                                           vectorizer,
                                           chunk_size,
                                           starting_chunk,
                                           ending_chunk,
                                           gpt_input):
                        print('Chunk_size: ' + str(chunk.shape[0]))
                        model.fit(chunk[:, :gpt_input], tf.reshape(chunk[:, -1], (-1, 1)), batch_size = batch_size, epochs=1)
                        print("Chunk Number " + str(chunk_number) + "/" +str(ending_chunk - starting_chunk + 1) + " processed!")
                        chunk_number += 1
        
    
    # Print the output of the Model
    output_seq = generate_output(gpt_input = gpt_input, model = model, vectorizer = vectorizer, text_size = output_length, input_sequence = [])
        
    if save == True and TPU == False:
        print('Saveeeeee')
        
        save_module(save_weights, model, vectorizer, save_tokenizer)
        
    if save == True and TPU == True:
        
        return save_weights, model, vectorizer, save_tokenizer, output_seq
        # Save the GPT Model
        #with open(save_weights, 'wb') as file:
        #    pickle.dump(model.weights, file)
        
        #Save the Vectorizer Model
        #vocabulary = vectorizer.get_vocabulary()

        # Encode the vocabulary as JSON-compatible strings
        #encoded_vocabulary = [word.encode('unicode_escape').decode('utf-8') for word in vocabulary]
        #encoded_vocabulary = encoded_vocabulary[2:]

        # Save the encoded vocabulary to a JSON file
        #with open(save_tokenizer, 'w') as f:
        #    json.dump(encoded_vocabulary, f)
        #    print("Vocabulary size saved: " + str(len(encoded_vocabulary)))
            
       
    if return_model_and_vectorizer:
        return model, vectorizer
    elif return_model_and_vectorizer_and_output:
        return model, vectorizer, output_seq.replace('@@ ', '')
    else:
        return output_seq.replace('@@ ', '')



# Example code to execute when the script file is called

def main():
    print("This code is executed when the script file is called directly.")

# Check if the script is being run as the main module
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', help='File: Corresponding to corpus or training text [String]')
    parser.add_argument('-l', '--learning-rate', help='Float: Learning Rate. The model will train ONLY IF the rate is > 0, skip otherwise [Float]', type=float)
    parser.add_argument('-ol', '--output-length', help='Length of the output sequence to be generated', type=int)
    parser.add_argument('-e', '--epochs', help='Number of training Epochs [Int]', type=int)
    parser.add_argument('-b', '--batch-size', help='Size of each batch [Int]', type=int)
    parser.add_argument('-s', '--gpt-input', help='Number of Tokens of text the model inputs at a time [Int]', type=int)
    parser.add_argument('-dm', '--d-model', help='Embedding layer output dimensions [Int]', type=int)
    parser.add_argument('-p', '--multi-head', help='Number of Multi-head Attention layer in parallel [Int]', type=int)
    parser.add_argument('-ds', '--decoder-stacks', help='Number of stacked Decoder layer [Int]', type=int)
    parser.add_argument('-sc', '--chunk-start', help='The chunk number in the corpus to mark it as the starting point of the training [Int]', type=int)
    parser.add_argument('-ec', '--chunk-end', help='The chunk number in the corpus to mark it as the end point of the training [Int]', type=int)
    parser.add_argument('-csz', '--chunk-size', help='The size of each chunk in KB.', type=int)
    parser.add_argument('-vs', '--vocabulary-start', help='Token number from the corpus to mark the starting point of vocabulary data [Int]', type=int)
    parser.add_argument('-ve', '--vocabulary-end', help='Token number from the corpus to mark the end point of vocabulary data [Int]', type=int)
    parser.add_argument('-sd', '--save', help='Save the Model and Vectorizer data to disk [True/False]', action='store_true')
    parser.add_argument('-lt', '--load-tokenizer', help='File: Vectorization layer [File]')
    parser.add_argument('-lw', '--load-weights', help='File: Model Weights [File]')
    parser.add_argument('-st', '--save-tokenizer', help='File: Saving Vectorizer File [File]')
    parser.add_argument('-sw', '--save-weights', help='File: Saving Model Weights[File]')
    parser.add_argument('-ot', '--optimizer', help='Optimizer consistent to TensorFlow optimizer class [tf.keras.optimizers]')
    parser.add_argument('-i', '--inference-only', help='Only Print the output of the model in Inference Mode [True/False]', action='store_true')
    parser.add_argument('-mv', '--model-vectorizer', help='Return Model, Vectorizer Tuple [True/False]', action='store_true')
    parser.add_argument('-mvo', '--model-vectorizer-output', help='Return Model, Vectorizer, Output Tuple [True/False]', action='store_true')
    parser.add_argument('-ga', '--gpt-style-attention', help='Uses GPT-styled attention. Note: (d-model) parameter should be divisible by (multi-head), otherwise the program will throw an error! [True/False]', action='store_true')
    parser.add_argument('-tpu', '--TPU', help='Use Tensor Processor Units (Distributed Learning)', action='store_true')
    
    
    args = parser.parse_args()
    
    
    data_path = args.data_path
    learning_rate = args.learning_rate
    output_length = args.output_length
    epochs = args.epochs
    batch_size = args.batch_size
    gpt_input = args.gpt_input
    d_model = args.d_model
    h = args.multi_head
    stacks = args.decoder_stacks
    chunk_start = args.chunk_start
    chunk_end = args.chunk_end
    chunk_size = args.chunk_size
    vocabulary_start = args.vocabulary_start
    vocabulary_end = args.vocabulary_end
    save = args.save
    load_tokenizer = args.load_tokenizer
    load_weights = args.load_weights
    save_tokenizer = args.save_tokenizer
    save_weights = args.save_weights
    optimizer = args.optimizer
    inference_only = args.inference_only
    model_and_vectorizer = args.model_vectorizer
    GPT_attention = args.gpt_style_attention
    model_vectorizer_output = args.model_vectorizer_output
    
    
    
    configuration = {
    'data_path': args.data_path,
    'learning_rate': args.learning_rate,
    'output_length': args.output_length,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'gpt_input': args.gpt_input,
    'd_model': args.d_model,
    'h': args.multi_head,
    'stacks': args.decoder_stacks,
    'chunk_start': args.chunk_start,
    'chunk_end': args.chunk_end,
    'chunk_size': args.chunk_size,
    'vocabulary_start': args.vocabulary_start,
    'vocabulary_end': args.vocabulary_end,
    'save': args.save,
    'load_tokenizer': args.load_tokenizer,
    'load_weights': args.load_weights,
    'save_tokenizer': args.save_tokenizer,
    'save_weights': args.save_weights,
    'optimizer': args.optimizer,
    'inference_only': args.inference_only,
    'model_and_vectorizer': args.model_vectorizer,
    'model_vectorizer_output': args.model_vectorizer_output,
    'GPT_Attention' : args.gpt_style_attention
    }

    # Save the configuration to a JSON file
    with open('last-configuration.json', 'w') as file:
        json.dump(configuration, file)
        
    
    
    if args.TPU == True:
        
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        
        
        strategy = tf.distribute.TPUStrategy(resolver)
        
        with strategy.scope():
            
            output = MinimalGPT(data_path = data_path, 
                   learning_rate = learning_rate, 
                   output_length = output_length,
                   epochs = epochs, 
                   batch_size = batch_size, 
                   gpt_input = gpt_input, 
                   d_model = d_model, 
                   h = h, 
                   decoder_stacks = stacks, 
                   starting_chunk = chunk_start,
                   ending_chunk = chunk_end,
                   chunk_size = chunk_size,
                   vocabulary_start = vocabulary_start,
                   vocabulary_end = vocabulary_end,
                   save = save, 
                   load_tokenizer = load_tokenizer, 
                   load_weights = load_weights, 
                   save_tokenizer = save_tokenizer,
                   save_weights = save_weights,
                   optimizer = optimizer,
                   inference_only = inference_only,
                   return_model_and_vectorizer = model_and_vectorizer,
                   return_model_and_vectorizer_and_output = model_vectorizer_output,
                   GPT_attention = GPT_attention, 
                   TPU = True)
    
        save_module(output[0], output[1], output[2], output[3])
        
        print(output[4])
        sys.exit(0)
    
    
    output = MinimalGPT(data_path = data_path, 
                       learning_rate = learning_rate, 
                       output_length = output_length,
                       epochs = epochs, 
                       batch_size = batch_size, 
                       gpt_input = gpt_input, 
                       d_model = d_model, 
                       h = h, 
                       decoder_stacks = stacks, 
                       starting_chunk = chunk_start,
                       ending_chunk = chunk_end,
                       chunk_size = chunk_size,
                       vocabulary_start = vocabulary_start,
                       vocabulary_end = vocabulary_end,
                       save = save, 
                       load_tokenizer = load_tokenizer, 
                       load_weights = load_weights, 
                       save_tokenizer = save_tokenizer,
                       save_weights = save_weights,
                       optimizer = optimizer,
                       inference_only = inference_only,
                       return_model_and_vectorizer = model_and_vectorizer,
                       return_model_and_vectorizer_and_output = model_vectorizer_output,
                       GPT_attention = GPT_attention,
                       TPU = False)
    print(output)
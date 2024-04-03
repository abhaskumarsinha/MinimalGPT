# ⚠️ All the support for MinimalGPT has ended, and is depreciated! Use [Corpus2GPT](https://github.com/abhaskumarsinha/Corpus2GPT) in the near future!
https://github.com/abhaskumarsinha/Corpus2GPT

# MinimalGPT: The 'Tiniest and Simplest GPT Model'
<img src='https://badgen.net/badge/license/MIT/blue'/> <img src='https://badgen.net/badge/stable/2.0.0/green?icon=github'/> <img src='https://badgen.net/badge/python/3.10/green'/> <img src='https://badgen.net/badge/Python-Script/TensorFlow/blue?icon=terminal'/> <img src='https://badgen.net/badge/AI-GPT/TensorFlow/purple?icon=terminal'/>


<img src="https://user-images.githubusercontent.com/31654395/236275892-300762b4-0640-412a-b9f5-066d748499f2.png" alt="MinimalGPT Logo" width="20%">


[[`GPT-1 Paper`](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)] [[`1002 short stories from project guttenberg`](https://www.kaggle.com/datasets/shubchat/1002-short-stories-from-project-guttenberg)] [[`logo.com`](https://wwww.logo.com/)] [[`Transformer - Paper`](https://arxiv.org/abs/1706.03762)] [[`Huggingface Transformers`](https://huggingface.co/docs/transformers/index)] [[`TensorFlow`](https://www.tensorflow.org/)] [[`BPE Tokenizer: subword-nmt`](https://github.com/rsennrich/subword-nmt)]

<p align='justify'>MinimalGPT is a concise, adaptable, and streamlined code framework that encompasses the essential components necessary for the construction, training, inference, and fine-tuning of the GPT model. This framework is implemented exclusively using Keras and TensorFlow, ensuring compatibility and coherence within the broader deep learning ecosystem.</p>
<h2>
  <p align='center'>NEW: CPU/GPU/TPU Support and support for loading big file datasets!</p>
</h2>

<h2> Code Specifications </h2>
<p align='justify'>
In the repository, we introduce two integral files that comprise our proposed framework. The first file, <i>GPT.py</i>, serves as the fundamental framework and encompasses crucial components such as blocks and layers. These components encompass multi-head attention, feedforward mechanisms, scaled dot product attention, positional encoding, softmaxed output, and an inference function for model prediction. The second file, <i>MinimalGPT.py</i>, streamlines the utilization of our framework by offering a concise command-line interface. This interface enables users to effortlessly perform essential operations, including model creation, training, saving, loading, fine-tuning, and inference, all condensed into a single command line execution. Furthermore, the files can be conveniently imported into Python code, allowing users to seamlessly incorporate them into their projects through a simple function call.
</p>
<h3> Requirements </h3>
Run the following command to install the required dependencies from the requirements.txt file:
<pre><code>
pip install -r requirements.txt
</code></pre>

<h3> Usage </h3>

<p align='justify'>
The model architecture is governed by several critical parameters, including <i>GPT_INPUT, D_MODEL, MULTI_HEAD</i>, and <i>DECODER_STACKS</i>. It is imperative to ensure consistency in these parameters to prevent issues related to loading the model for subsequent re-training or inference processes. In situations where uncertainty arises, referring to the configuration file generated during the previous run can provide valuable insights. Furthermore, the <i>VOCABULARY_START</i> and <i>VOCABULARY_END</i> parameters play a crucial role in defining the window markers for the corpus. These markers aid in generating the Vectorizer layer, which extracts the vocabulary from the corpus within the specified START and END token counts. It is essential to note that tokens within the corpus are separated by whitespaces, and the inclusion of <i>VOCABULARY_START</i> and <i>VOCABULARY_END</i> becomes especially relevant when a token file is not explicitly specified.

Also, note that BOTH - tokenizer file as well as weights of the model are saved/loaded at a time. Currently the code doesn't supports saving/loading these two files separatedly.

The inference mode (-i) doesn't only requires model parameters and saved tokenizer and weights file to generate inference data. It should be used with (-ol) switch.
</p>


<pre><code>
usage: MinimalGPT.py [-h] [-d DATA_PATH] [-l LEARNING_RATE]
                     [-ol OUTPUT_LENGTH] [-e EPOCHS] [-b BATCH_SIZE]
                     [-s GPT_INPUT] [-dm D_MODEL] [-p MULTI_HEAD]
                     [-ds DECODER_STACKS] [-ts TOKEN_START] [-te TOKEN_END]
                     [-vs VOCABULARY_START] [-ve VOCABULARY_END] [-sd]
                     [-lt LOAD_TOKENIZER] [-lw LOAD_WEIGHTS]
                     [-st SAVE_TOKENIZER] [-sw SAVE_WEIGHTS] [-ot OPTIMIZER]
                     [-i] [-mv] [-mvo]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data-path DATA_PATH
                        File: Corresponding to corpus or training text
                        [String]
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Float: Learning Rate. The model will train ONLY IF the
                        rate is > 0, skip otherwise [Float]
  -ol OUTPUT_LENGTH, --output-length OUTPUT_LENGTH
                        Length of the output sequence to be generated
  -e EPOCHS, --epochs EPOCHS
                        Number of training Epochs [Int]
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Size of each batch [Int]
  -s GPT_INPUT, --gpt-input GPT_INPUT
                        Number of Tokens of text the model inputs at a time
                        [Int]
  -dm D_MODEL, --d-model D_MODEL
                        Embedding layer output dimensions [Int]
  -p MULTI_HEAD, --multi-head MULTI_HEAD
                        Number of Multi-head Attention layer in parallel [Int]
  -ds DECODER_STACKS, --decoder-stacks DECODER_STACKS
                        Number of stacked Decoder layer [Int]
  -ts TOKEN_START, --token-start TOKEN_START
                        The token number in the corpus to mark it as the
                        starting point of the training [Int]
  -te TOKEN_END, --token-end TOKEN_END
                        The token number in the corpus to mark it as the end
                        point of the training [Int]
  -vs VOCABULARY_START, --vocabulary-start VOCABULARY_START
                        Token number from the corpus to mark the starting
                        point of vocabulary data [Int]
  -ve VOCABULARY_END, --vocabulary-end VOCABULARY_END
                        Token number from the corpus to mark the end point of
                        vocabulary data [Int]
  -sd, --save           Save the Model and Vectorizer data to disk
                        [True/False]
  -lt LOAD_TOKENIZER, --load-tokenizer LOAD_TOKENIZER
                        File: Vectorization layer [File]
  -lw LOAD_WEIGHTS, --load-weights LOAD_WEIGHTS
                        File: Model Weights [File]
  -st SAVE_TOKENIZER, --save-tokenizer SAVE_TOKENIZER
                        File: Saving Vectorizer File [File]
  -sw SAVE_WEIGHTS, --save-weights SAVE_WEIGHTS
                        File: Saving Model Weights[File]
  -ot OPTIMIZER, --optimizer OPTIMIZER
                        Optimizer consistent to TensorFlow optimizer class
                        [tf.keras.optimizers]
  -i, --inference-only  Only Print the output of the model in Inference Mode
                        [True/False]
  -mv, --model-vectorizer
                        Return Model, Vectorizer Tuple [True/False]
  -mvo, --model-vectorizer-output
                        Return Model, Vectorizer, Output Tuple [True/False]
</pre></code>

<h3> Examples </h3>


<h4> Example of Model Creation and Training </h4>

<p align='justify'>Assuming the desired model specifications entail GPT_INPUT = 10, D_MODEL = 128, MULTI_HEAD = 8, and DECODER_STACKS = 1, and the corpus token range for training spans from TOKEN_START = 0 to TOKEN_END = 40000, and generate the vectorizer layer from the corpus span from VOCABULARY_START = 0 to VOCABULARY_END = 200000, the following command is executed to initiate the model training process. The resulting weights and tokenizer data are saved in the designated folder. The subsequent outputs illustrate the outcome of this command execution.</p>

<pre><code>
PS C:\gpt> python MinimalGPT.py -d './dataset/output_dataset.txt' -l 0.001 -ol 200 -e 4 -b 512 -s 10 -dm 128 -p 8 -ds 1 -ts 0 -te 40000 -vs 0 -ve 200000 -sd -st './models/tokenizer.mgt' -sw './models/weights.mgw'
Total tokens: 40000
100%|██████████████████████████████████████████████████████████████████████████████| 200000/200000 [02:02<00:00, 1636.38it/s]
New Vectorizer created successfully...
Vocabulary Size: 14270
100%|██████████████████████████████████████████████████████████████████████████████| 39989/39989 [00:00<00:00, 302926.25it/s]
100%|█████████████████████████████████████████████████████████████████████████████| 39989/39989 [00:00<00:00, 1289942.19it/s]
(None, 10, 128)
Epoch 1/4
79/79 [==============================] - 88s 1s/step - loss: 7.8692
Epoch 2/4
79/79 [==============================] - 92s 1s/step - loss: 3.8066
Epoch 3/4
79/79 [==============================] - 93s 1s/step - loss: 1.1487
Epoch 4/4
79/79 [==============================] - 92s 1s/step - loss: 0.2900
100%|██████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:05<00:00, 34.70it/s]
Vocabulary size saved: 14270
         and her eyes in the library. She was the rather large woman, although not fat, and when she wore high heels--which sh
e was not prone to do, because although Cutter would not have cared, she kept trying to project into other people's minds and
trying, as she said, "Not to do anything to them, that I wouldn't want them to do you me."--she rose a good inch above Cutter.
 She was pleasant humored, and cooperative, and the one great irritant about her that annoyed Cutter, was the fact that she wa
s not capable of meeting life wholeheartedly and with strength. She steadily worried about other people's feelings and thought
s, so that Cutter wondered if she were capable of the slightest personal conviction. Yet that weakness was an advantage at the
 same time, to him, because she worked constantly toward making him happy. The house was run to his minutest liking, and the s
ervants liked her, so that while she did not use a strong enough
</code></pre>

<h4> Fine-tuning </h4>

<p align='justify'>Suppose we want to fine-tune the above model (or retrain it), then the command to re-load the tokenizer and weights and retrain it on a new text of a specified window range of the corpus is given below:</p>

<pre><code>
PS C:\gpt> python MinimalGPT.py -d './dataset/output_dataset.txt' -l 0.00005 -ol 200 -e 1 -b 512 -s 10 -dm 128 -p 8 -ds 1 -ts 80000 -te 120000 -sd -st './models/tokenizer2.mgt' -sw './models/weights2.mgw' -lt './models/tokenizer.mgt' -lw './models/weights.mgw'
Total tokens: 40000
100%|██████████████████████████████████████████████████████████████████████████████| 39989/39989 [00:00<00:00, 302923.51it/s]
100%|█████████████████████████████████████████████████████████████████████████████| 39989/39989 [00:00<00:00, 1428099.68it/s]
(None, 10, 128)
79/79 [==============================] - 81s 993ms/step - loss: 7.9725
100%|██████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:06<00:00, 30.29it/s]
Vocabulary size saved: 14270
         of her own the black of my own and my wife had could seen the house at the same moment her mind caught the first sugg
estion of the folded paper. “But he must have a name! Where is the paper?” She moved to the desk, and began to turn over the s
cattered documents that littered it. The first that caught her eye was an unfinished letter in her husband’s hand, with his pe
n lying across it, as though dropped there at a sudden summons. “My dear Parvis,”--who was Parvis?--“I have just received your
 letter announcing Elwell’s death, and while I suppose there is now no farther risk of trouble, it might be safer--” That was
all. The “risk of trouble” was easily explained by the newspaper clipping which had apprised Mary of the suit brought against
her husband by one of his associates in the Blue Star enterprise. The only new information conveyed in the letter was the fact
 of its showing Boyne,
</code></pre>

<h4> Inference Mode </h4>
<p align='justify'>The inference mode involves the loading of pre-trained weights and vectorizer. These components are then utilized to execute the model, generating outputs of a specified length as specified.</p>

<code><pre>
PS C:\gpt> python MinimalGPT.py -i -ol 500 -e 6 -b 512 -s 10 -dm 128 -p 8 -ds 1 -lt './models/tokenizer2.mgt' -lw './models/weights2.mgw'
(None, 10, 128)
100%|██████████████████████████████████████████████████████████████████████████████████████| 490/490 [00:13<00:00, 35.93it/s]
         of her own “on the other from the inel’--a little sensational, of course. But I guess you’d better look it over.” He
held out a newspaper to Mary, who unfolded it slowly, remembering, as she did so, the evening when, in that same room, the per
usal of a clipping from the “Sentinel” had first shaken the depths of her security. As she opened the paper, her eyes, shrinki
ng from the glaring head-lines, “Widow of Boyne’s Victim Forced to Appeal for Aid,” ran down the column of text to two portrai
ts inserted in it. The first was her husband’s, taken from a photograph made the year they had come to England. It was the pic
ture of him that she liked best, the one that stood on the writing-table up-stairs in her bedroom. As the eyes in the photogra
ph met hers, she felt it would be impossible to read what was said of him, and closed her lids with the sharpness of the pain.
 “I thought if you felt disposed to put your name down--” she heard Parvis continue. She opened her eyes with an effort, and t
hey fell on the other portrait. It was that of a youngish man, slightly built, in rough clothes, with features somewhat blurre
d by the shadow of a projecting hat-brim. Where had she seen that outline before? She stared at it confusedly, her heart hamme
ring in her throat and ears. Then she gave a cry. “This is the man--the man who came for my husband!” She heard Parvis start t
o his feet, and was dimly aware that she had slipped backward into the corner of the sofa, and that he was bending above her i
n alarm. With an intense effort she straightened herself, and reached out for the paper, which she had dropped. “It’s the man!
 I should know him anywhere!” she cried in a voice that sounded in her own ears like a scream. Parvis’s voice seemed to come t
o her from far off, down endless, fog-muffled windings. “Mrs. Boyne, you’re not very well. Shall I call somebody? Shall I get
a glass of water?” “No, no, no!” She threw herself toward him, her hand frantically clenching the newspaper. “I tell you, it’s
 the man! I KNOW him! He spoke to me in the garden!” Parvis took the journal from her, directing his glasses to the portrait.
“It can’t be, Mrs. Boyne. It’s Robert Elwell.” “Robert Elwell?” Her white
</pre></code>



<h4> Importing the model into a project </h4>

<p align='justify'>Incorporating the trained models generated through the utilization of MinimalGPT.py into your project is a straightforward process facilitated by importing the MinimalGPT function and configuring it according to the desired specifications. This can be achieved by setting the parameters return_model_and_vectorizer = True or return_model_and_vectorizer_and_output = True within the inference_only = True (Inference Mode) framework. Additionally, the training, creation, and exportation of the model can be accomplished using a similar approach, paralleling the command-line mode. For a comprehensive illustration of these procedures, the accompanying <a href='https://github.com/abhaskumarsinha/MinimalGPT/blob/main/examples/models.ipynb'>Jupyter Notebook</a> provides an exemplar demonstration.</p>
<code><pre>
from MinimalGPT import MinimalGPT


model = MinimalGPT(output_length = 200, gpt_input = 10, d_model = 128, h = 8, decoder_stacks = 1, load_tokenizer = './models/tokenizer3.mgt', load_weights = './models/weights3.mgw', inference_only = True, return_model_and_vectorizer_and_output = True)
model[0].summary()

_________________________________________________________________
Model: "model"
_________________________________________________________________

 Layer (type)                Output Shape              Param  
 
=================================================================
 input_1 (InputLayer)        [(None, 10)]              0         
                                                                 
 embedding (Embedding)       (None, 10, 128)           1826816   
                                                                 
 positional_embedding (Posit  (None, 10, 128)          0         
 ionalEmbedding)                                                 
                                                                 
 decoder (Decoder)           (None, 10, 128)           37160     
                                                                 
 flatten (Flatten)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 14273)             18283713  
                                                                 
 tf.nn.softmax (TFOpLambda)  (None, 14273)             0         
                                                                 
=================================================================
Total params: 20,147,689
Trainable params: 20,147,689
Non-trainable params: 0
_________________________________________________________________
</pre></code>

<h2> Implementation Specifications </h2>
<p align='justify'>The model implemented here differs a little bit in comparision to the original paper implementation. The matrix formed after concatenating the heads of the scaled dot-product output is multiplied by the matrix parameter of size key dimension x d_model. For practical purpose, this little tweak to reduce the number of parameter would lead to a little bit increase in performance due to trainable parameter optimization.</p>


<h2> Results </h2>
<i>Follow the <a href='https://github.com/abhaskumarsinha/MinimalGPT/tree/main/examples'>example folder</a> for Notebooks containing the samples.</i>

<h2> Troubleshooting </h2>
Feel free to open tickets in the issue tab in case you encounter any error or have any specific feature request in mind.


<h2> References/Further Reading </h2>


1. Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
2. Radford, Alec, et al. "Improving language understanding by generative pre-training." (2018).
3. Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
4. Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.
5. Howard, Jeremy, and Sebastian Ruder. "Universal language model fine-tuning for text classification." arXiv preprint arXiv:1801.06146 (2018).
6. Petroni, Fabio, et al. "Language models as knowledge bases?." arXiv preprint arXiv:1909.01066 (2019).


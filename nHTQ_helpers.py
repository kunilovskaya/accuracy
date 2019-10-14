## this script contains functions to be used in *.py and *ipynb that use TF-keras to learn Human Translation Quality (HTQ)
import os
import pandas as pd
import gensim
import keras.backend as K

from keras.layers import Dense, Input, LSTM, Bidirectional, Lambda, concatenate
from keras.models import Model
from keras.layers.merge import dot
from keras.layers import Embedding


def preparedata(directory):
    ourdic = []
    print('Collecting data from the files...')
    for subdir in os.listdir(directory):
        files = [f for f in os.listdir(os.path.join(directory, subdir)) if f.endswith('.conllu')]
        for f in files:
            rowdic = {'doc': f.strip(), 'group': subdir}
            doc = open(os.path.join(directory, subdir, f))
            text = doc.read().strip() #.replace('\n', ' ')
            doc.close()
            rowdic['text'] = text
            ourdic.append(rowdic)
    ourdic = pd.DataFrame(ourdic)
    return ourdic

def load_embeddings(embeddings_file):
    # Определяем формат модели по её расширению:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):  # Бинарный формат word2vec
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):  # Текстовый формат word2vec
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='ignore') ##unicode_errors='replace'
    else:  # Нативный формат Gensim?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)
    emb_model.init_sims(replace=True)  # normalise vectors just in case
    return emb_model

def get_index(word, vocabulory=None):
    ## get word index in the models vocabulary
    if word in vocabulory:
        return vocabulory[word].index
    ## set index to OOV items like putin_ADJ to 0
    else:
        return 0
    
## see https://github.com/keras-team/keras/issues/5541
## https://gist.github.com/GKarmakar/3aa0c643ddb0688a9bfc44b43b84edd8

## K.exp = math_ops.exp(x)???
## https://www.tensorflow.org/api_docs/python/tf/math/exp: Computes exponential of x element-wise: y= e^x
## about e=2.7182818 (Euler's number) https://www.mathsisfun.com/numbers/e-eulers-number.html
## a negative exponent can be expressed as its positive reciprocal: 5^-3 = 1/5^3, so the result of my function is either pos or neg, but always smaller than -1:1
def exponent_neg_manhattan_distance(source, target):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(source - target), axis=1, keepdims=True))


def exponent_neg_euclidean_distance(source, target):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    ## K.exp() is a constant e to the power of -x, where x is the distance; the purpose of it is to put the result into the range of -1;1
    return K.exp(-K.sqrt(K.sum(K.square(source - target), axis=1, keepdims=True)))

## construct a functional model
def createmodel(loss='binary_crossentropy', mode='concate', units=32, seq_length=None, en_embeds=None, ru_embeds=None, classes=None):
    ## encoder
    en_input = Input(shape=(seq_length,), name='ST_wsequences')
    ## decoder
    ru_input = Input(shape=(seq_length,), name='TT_wsequences')
    
    ## represent words in texts with their embeddings (what happens to OOVwords with index=0)
    en_vectors = en_embeds.vectors
    ru_vectors = ru_embeds.vectors
    print('The shape of the input vectors for each language:\nEN %s\nRU %s' % (en_vectors.shape, ru_vectors.shape))
    ## Input shape: 2D tensor with shape: (batch_size, sequence_length)
    ## Output shape: 3D tensor with shape: (batch_size, sequence_length, output_dim)
    en_emb = Embedding(input_dim=en_vectors.shape[0], output_dim=en_vectors.shape[1], weights=[en_vectors],
                       input_length=seq_length, trainable=False, name='en_embeddings')(en_input)
    ru_emb = Embedding(input_dim=ru_vectors.shape[0], output_dim=ru_vectors.shape[1], weights=[ru_vectors],
                       input_length=seq_length, trainable=False, name='ru_embeddings')(ru_input)
    # units is a hyperparameter; the output of the lstm layer
    # this is where the model would process the input sequence in both directions
    
    # To share a layer across different inputs, simply instantiate the layer once, then call it on as many inputs as you want
    ## # When we reuse the same layer instance, the weights of the layer are also being reused
    ## (it is effectively *the same* layer)
    shared_lstm = Bidirectional(LSTM(units=units, name="biLSTM"))
    
    en_x = shared_lstm(en_emb)
    ru_x = shared_lstm(ru_emb)
    
    # Calculates the pair-wise distance between ST and TT; the output should be length 542
    if mode == 'dist_manh':
        res = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                     output_shape=lambda x: (x[0][0], 1))([en_x, ru_x])
    if mode == 'dist_euc':
        res = Lambda(function=lambda x: exponent_neg_euclidean_distance(x[0], x[1]),
                     output_shape=lambda x: (x[0][0], 1))([en_x, ru_x])
        print('Using %s to compare text vectors pair-wise and get %s-long input (of type %s) to the classifier' % (
            mode.upper(), res.shape, type(res)))
        print()
    
    if mode == 'sim_dot':
        res = dot([en_x, ru_x], axes=1, name='Cosine')
    
        print('Using %s to compare text vectors pair-wise and get %s-long input (of type %s) to the classifier' % (
        mode.upper(), res.shape, type(res)))
        print()
    
    ## instead of similarity, use concatenation of the text vectors output by LSTMs:
    if mode == 'concate':
        res = concatenate([en_x, ru_x], axis=1, name='Concatenated')
        print('The shape of the concatenated input to Dense', res.shape)
    
    output = Dense(classes, activation='softmax', name='Output')(res)
    
    ## compile the model
    model = Model(inputs=[en_input, ru_input], outputs=output)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    
    # print the model architecture
    print(model.summary())
    
    return model
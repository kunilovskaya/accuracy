# coding: utf-8

# this script is an attempt to classify good and bad translations from RusLTC
# on bilingual vectorised representations of lempos (no stopwords) in sources and targets
# supposedly in this set up the algorithm captures accuracy of the translation rather than (morphosyntactic) fluency

## we try several ideas here:
# arrange the separate inputs for ST and TT, learn their respective text vectors and either
# (1) concatenate the vectors and use the resulting text-pair vector as the classifier input
# or
# (2) calculate a similarity measure between the two text vectors and use its value for each text pair as a single feature in the classifier

## this is supposed to ensure reproducible results at each run of the script
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import time
import numpy as np

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# TensorFlow and tf.keras
import logging
import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras import preprocessing

from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

print(tf.__version__)

## import the functions from the helper scripts
from nHTQ_helpers import preparedata, load_embeddings, get_index, createmodel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

## preprosessing of input texts with from https://github.com/akutuzov/webvectors/blob/master/preprocessing/rus_preprocessing_udpipe.py
datadir = '/home/masha/venv/data/lempos/'  # The path to where subdirectories with text files are
df = preparedata(datadir)
print('Look at the data loaded in a df:')
print(df.shape)
print(df.head())

# transform the df to align each target with its source: doc, src, tgt, label
targets_df = df[~(df['group']=='source')]
sources_df = df[df['group']=='source']

aligned = targets_df.copy()
aligned.insert(loc=2, column='source', value=None)
aligned.columns = ['doc', 'group', 'source', 'target']

aligned['temp'] = aligned['doc'].replace(to_replace='RU',value=r'EN', regex=True)
aligned['temp'] = aligned['temp'].replace(to_replace='_\d+\.conllu',value=r'.conllu', regex=True)


sfns = aligned['temp'].tolist()
for i in sfns:
## two ways of putting the value from one column (text) in one df on the value in the other column (temp filename) into the specified column (source) of the other df, following the list respecting the order of the cells 
#     aligned.loc[aligned.temp == i, 'source'] = sources_df[sources_df['doc']=='EN_1_101.conllu']['text'].values[0]
    aligned.loc[aligned.temp == i, 'source'] = sources_df.loc[sources_df['doc'] == i, 'text'].item()
print('This is the same data aligned at text level:')
print(aligned.shape)
print(aligned.head())

logger.info('===Preparing our Y====')

yy_train = aligned['group'].tolist()

classes = sorted(list(set(aligned['group'].tolist())))
num_classes = len(classes)
logger.info('Number of classes: %d' % num_classes)
print('===========================')
print('Distribution of classes in the dataset:')
print(aligned.groupby('group').count())
print('===========================\n')

# Convert class labels into indices
y_train0 = [classes.index(i) for i in yy_train]

## Convert indices to categorical values to use with (binary_ or categorical_)crossentropy loss
y_train = to_categorical(y_train0, num_classes)
print('Class labels converted to binary categorical', y_train[:5])
print('We have a binary classification (see number of columns):', y_train.shape)

logger.info('===Preparing our enX and ruX====')
## encode ST and TT with the respective bilingual embeddings
src_path = '/home/masha/MUSE/rig1_res/vectors-en.vec'
tgt_path = '/home/masha/MUSE/rig1_res/vectors-ru.vec'

en_embeddings = load_embeddings(src_path)
ru_embeddings = load_embeddings(tgt_path)

logger.info('Loading word vectors')

en_voc = en_embeddings.vocab
ru_voc = ru_embeddings.vocab
print('Look at the embeddings vocabularies for both languages')
print(list(en_voc.keys())[0:5])
print(list(ru_voc.keys())[0:5])
print('First source text:', aligned['source'][0].split()[:20])
print('\nThe first of the aligned translations:', aligned['target'][0].split()[:20])

##convert them to embedding indices in a joint biling embedding file
entrain0 = [[get_index(w,en_voc) for w in text.split()] for text in aligned['source']]
rutrain0 = [[get_index(w,ru_voc) for w in text.split()] for text in aligned['target']]

## here we are getting rid of OOV items:
entrain = [[id for id in text if id != 0] for text in entrain0]
rutrain = [[id for id in text if id != 0] for text in rutrain0]

print('First source text:', entrain0[0][:20])
print('First source text with no OOV (tot num of enOOV=123146! ruOOV=71558):', entrain[0][:20])

logger.info('Average SOURCE text length: %s tokens'
                % "{0:.1f}".format(np.mean(list(map(len, entrain))), 1)) #aligned['source'].str.split()
logger.info('Average TARGET text length: %s tokens'
                % "{0:.1f}".format(np.mean(list(map(len, rutrain))), 1))
logger.info('Maximum SOURCE text length: %s tokens'
                % "{0:.1f}".format(np.max(list(map(len, entrain))), 1))
logger.info('Maximum TARGET text length: %s tokens'
                % "{0:.1f}".format(np.max(list(map(len, rutrain))), 1))

# Padding: make all texts lengths equal to absolute max in the data by filling in zeros
max_seq_length = 685  
vectorized_en = preprocessing.sequence.pad_sequences(
    entrain, maxlen=max_seq_length, truncating='post', padding='post')
vectorized_ru = preprocessing.sequence.pad_sequences(
    rutrain, maxlen=max_seq_length, truncating='post', padding='post')
print('The translation turned into %s ...,\n its length is %s' % (vectorized_ru[0][:5], len(vectorized_ru[0]))) #, 
print('Vextorised (ru) data parameters:', vectorized_ru.shape)

## 'binary_crossentropy', 'categorical_crossentropy'
## 'concate', 'sim_dot', 'dist_manh', 'dist_euc'

## run the function which compiles the model
model = createmodel(loss='binary_crossentropy', mode='dist_euc', units=32, seq_length=max_seq_length, en_embeds=en_embeddings, ru_embeds=ru_embeddings, classes=num_classes)

# Train the model on our data, use stratified train-test split for validation and testing, and optionally calculated class weights

## compute the class weights to pass to the scoring function which generates the update weights (values)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train0),
                                                 y_train0)
## convert class_weights to dict
class_weight_dict = dict(enumerate(class_weights))
print('class weights calculated from the distribution:',class_weight_dict)

earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='max')

start = time.time()
## batch_size: number of samples per gradient update.
## verbosity: 0 = silent, 1 = progress bar, 2 = one line per epoch.
## validation_split: evaluate the loss and any model metrics on this data at the end of each epoch
## shuffle: whether to shuffle the training data before each epoch
## class_weight: "pay more attention" to samples from an under-represented class; https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
## {0: 2., 1: 1.} "treat every instance of class 0 (bad) as 2 instances of class 0" > the loss becomes a weighted average

## make sure that the testset held out for testing each epoch and evaluating the model has a proportional distribution of each class samples
enX_train, enX_test, Y_train, Y_test = train_test_split(vectorized_en, y_train,
                                                    stratify=y_train, 
                                                    test_size=0.1, random_state=42)
# print(Y_test[:5])
ruX_train, ruX_test, Y_train0, Y_test0 = train_test_split(vectorized_ru, y_train,
                                                    stratify=y_train, 
                                                    test_size=0.1, random_state=42)
# print('test that Ys are the same': Y_test, Y_test)
## iterating on the data in batches of 4 samples
history = model.fit([enX_train,ruX_train], Y_train, epochs=10, verbose=1, validation_data=([enX_test,ruX_test],Y_test),
         batch_size=1, shuffle=True, class_weight=class_weight_dict, callbacks=[earlystopping]) ## class_weight=class_weight_dict, None

end = time.time()
training_time = int(end - start)
logger.info('Trained in %s minutes' % str(round(training_time/60)))


logger.info('Evaluating the model:')
print(model.metrics_names)
score = model.evaluate([enX_test,ruX_test],Y_test, verbose=1)
logger.info('Loss function value: %s' % "{0:.4f}".format(score[0]))
logger.info('Accuracy on the test set: %s' % "{0:.4f}".format(score[1]))

# Используем функцию из sklearn чтобы посчитать F1 по каждому классу:
predictions = model.predict([enX_test,ruX_test])
predictions = np.around(predictions)  # проецируем предсказания модели в бинарный диапазон {0, 1}

# Конвертируем предсказания обратно из чисел в текстовые метки классов
y_test_real = [classes[np.argmax(pred)] for pred in Y_test]
predictions = [classes[np.argmax(pred)] for pred in predictions]

logger.info('Classification results on the test set:')
print(classification_report(y_test_real, predictions))
print("Confusion matrix\n", confusion_matrix(y_test_real, predictions))

fscore = precision_recall_fscore_support(y_test_real, predictions, average='macro')[2]
logger.info('Macro-F1 on the test set: %s' % "{0:.4f}".format(fscore))


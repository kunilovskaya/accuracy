# accuracy
non-translationese approaches to estimating human translation quality

## GOAL

predict text-level human translation quality

## DATA

542 hand-annotated targets in Russian to 105 English sources (all in the general domain of mass-media texts)

| labels |  words  | texts |
|--------|---------|-------|
| good   | 127,192 |  329  |
|--------|---------|-------|
|  bad   |  87,367 |  213  |
|--------|---------|-------|
| source |  46,218 |  105  |

Pre-trained bilingual embeddings (en, ru and a concatenated file en-ru) can be downloaded [here](https://dev.rus-ltc.org/static/misc/accuracy/biling_vector_models.tar.gz).
They were trained following the unsupervised approach suggested in [MUSE project](https://github.com/facebookresearch/MUSE) on *enwiki_upos_skipgram_300_5_2017* and *ruwikiruscorpora_upos_skipgram_300_2_2019* from WebVectores and RusVectores respectively.

## First Results

The initial experiments to explore the dataset are run with the following results:
*(Note that the macro F1 score for a stratified dummy classifier, which randomly assigns labels with the respect to the data distribution, over 10 folds: 0.478; accuracy: 0.514)*

### (a) 45 translationese morphosyntactic features: F1 = 0.6419
### (b) tf-idf scaled BOW:
* char, ngram_range=(3, 3): **F1 = 0.674**
* word, ngram_range=(3, 3): F1 = 0.652
* pos, ngram_range=(3, 3): Macro-F1 on the 0.3 test set: 0.3779)

## On lemmatized no-stop words embeddings transposed to the shared vector space
### (c) summed and concatenated raw biling embeddings for each text pair: F1 = 0.378 (no fine-tuning)
### (d) cosine similarity between summed word vectors of ST and TT as a single feature: F1 = 0.523

## Experiments involving Keras Neural models:
shared parameters and settings: 
* max text length for padding, 
* batch_size=1
* shuffle=True
* class_weight=class_weight_dict
* callbacks=[earlystopping], patience=5, min_delta=0.0001, mode='max'
* epochs=10
* evaluation: Macro-F1 on the 0.1 stratified test set
### (e) a classifier on the concatenated ST and TT vectors learnt by a shared Bidirectional LSTM from lempos of ST and TT (no stop words!), encoded with the respective bilingual embeddings (learned in MUSE): 
* **F1=0.6177 (with units=32)**; F1=0.5988 (with units=100)

### (f) similarity score between the outputs of two biLSTMs (one for ST, one for TT)
#### exponentiated negative metrics (e^-x)
* manhattan distance=euclidean distance (or all prediction for the non-majority class if class_weight is added): patience=5---F1=0.2857 patience=3---F1=0.3750
Confusion matrix for the testset
 0 22
 0 33
#### lose exponent_neg_ (K.exp(-X)):
manh F1=0.5623, but weird things are happening to loss reported in score[0]

* cosine similarity (implemented as dot product for normed vectors) **units=32 -- F1=0.6304**; units=50 -- F1=0.3750 (this is predicting the marjority class)

## On raw biling vectors, learnt from [ft monolingual cc embeddings](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md)





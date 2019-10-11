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
* char, ngram_range=(3, 3): F1 = 0.674
* word, ngram_range=(3, 3): F1 = 0.652
* pos, ngram_range=(3, 3): F1 = 1.00 (I can't find the error!, the same result is seen on 10fold cv, and on 0.3 train-test)
### (c) summed and concatenated embeddings for each text pair: F1 = 0.378
### (d) cosine similarity between ST and TT as a single feature: F1 = 0.523
### (e) Bidirectional LSTM on the concatenated lempos of ST and TT, represented with the respective bilingual embeddings (learned in MUSE) and padded to the max text length treated as instances: Macro-F1 on the test set: 0.6333
### (f) similarity score between the outputs of two biLSTMs (one for ST, one for TT)
exponentiated negative metrics (e^-x)
* manhattan distance=euclidean distance (or all prediction for the non-majority class if class_weight is added): F1 = 0.3750
Confusion matrix for the testset
 0 22
 0 33
* 1-cosine similarity: failed to implement
### (h) concatenated text vectors output by LSTMs: 0.3750 (no change!, same result for all sim measures and for concatenated text vectors from shared LSTM)


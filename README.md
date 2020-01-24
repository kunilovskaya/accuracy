# accuracy
non-translationese approaches to estimating human translation quality
this part focuses on accuracy as a component in TQ estimation
(see the complement [fluency project]())

## GOAL
use cross-linguistic text similarity to capture accuracy

## DATA
542 hand-annotated targets in Russian to 105 English sources (all in the general domain of mass-media texts)
The data comes from [RusLTC](https://www.rus-ltc.org/static/html/about.html), which sources quality-labeled translations from a number of translation competitions and university exam settings.
The translations were graded or ranked by either university translation teachers or by a panel of professional jurors in the contest settings. 


| labels |  words  | texts |
|--------|---------|-------|
| good   | 127,192 |  329  |
|--------|---------|-------|
|  bad   |  87,367 |  213  |
|--------|---------|-------|
| source |  46,218 |  105  |

Pre-trained bilingual embeddings (en, ru and a concatenated file en-ru) can be downloaded [here](https://dev.rus-ltc.org/static/misc/accuracy/biling_vector_models.tar.gz).
They were trained following the unsupervised approach suggested in [MUSE project](https://github.com/facebookresearch/MUSE) on *enwiki_upos_skipgram_300_5_2017* and *ruwikiruscorpora_upos_skipgram_300_2_2019* from WebVectores and RusVectores respectively.

## Results

The initial experiments to explore the dataset are run with the following results:
*(Note that the macro F1 score for a stratified dummy classifier, which randomly assigns labels with the respect to the data distribution, over 10 folds: 0.478; accuracy: 0.514)*

(a) 45 translationese morphosyntactic features: F1 = 0.6419
(b) tf-idf scaled BOW:
* char, ngram_range=(3, 3): **F1 = 0.674**
* word, ngram_range=(3, 3): F1 = 0.652
* pos, ngram_range=(3, 3): Macro-F1 on the 0.3 test set: 0.3779)

**On lemmatized no-stop words embeddings transposed to the shared vector space**
(c) summed and concatenated raw bilingual embeddings for each text pair: F1 = 0.607 (no fine-tuning)
(d) cosine similarity between summed word vectors of ST and TT as a single feature: F1 = 0.579

**Experiments involving Keras Neural models (and no averaging vectors for the whole texts):**
shared parameters and settings: 
* max text length for padding, 
* batch_size=1
* shuffle=True
* class\_weight=class\_weight\_dict
* callbacks=[earlystopping], patience=5, min_delta=0.0001, mode='max'
* epochs=10
* evaluation: Macro-F1 on the 0.1 stratified test set
(e) a classifier on the concatenated ST and TT vectors learnt by a shared Bidirectional LSTM from lempos of ST and TT (no stop words!), encoded with the respective bilingual embeddings (learned in MUSE): 
* **F1=0.6177 (with units=32)**; F1=0.5988 (with units=100)

(f) similarity score between the outputs of two `siamese` biLSTMs (one for ST, one for TT) with the following variants of the similarity metrics:
* with exponentiated negative metrics (e^-x)
1. manhattan distance=euclidean distance: patience=5---F1=0.2857 patience=3---F1=0.3750
Confusion matrix for the testset
 0 22
 0 33
* lose exponent_neg_ (K.exp(-X)):
1. manh F1=0.5623, but weird things are happening to loss reported in score[0]

* cosine similarity (implemented as dot product for normed vectors) **batch_size=1, units=128 -- F1=0.630**
* cosine similarity with best param from previous experiments, with the weird underdocumented "multi" embeddings pre-trained in MUSE: F1=0.3750

**NB! the results on vectors with functional words are a bit worse in all of the above settings!**

## On raw biling vectors, learnt from [ft monolingual cc embeddings](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md)
to be added

## Vectors evaluation
### monolingual vectors 
*Spearman's rho on enSimLex-999/ruSimLex-965*

|    vectors         |  en  |  ru  |
|--------------------|------|------|
| CommonCrawl tokens | .371 | .308 |
|--------------------|------|------|
| wiki lempos        | .401 | .321 |
|--------------------|------|------|
| wiki lempos func   | .413 | .311 |
|--------------------|------|------|
| wiki lempos func   |      |      |
| en2ru space        | .413 | .311 |
|--------------------|------|------|
| rnc5papers lempos  |      |      |
| func               |  --- | .315 |
|--------------------|------|------|

### bilingual vectors 
(English vectors mapped to the Russian semantic space by scripts from [MUSE project](https://github.com/facebookresearch/MUSE))
on a bilingual glossary (1.5K word pairs), contextual dissimilarity measure (csls\_knn), model=100K

|        vectors      |  P@1  |   P@5 |  P@10 |
|---------------------|-------|-------|-------|
| CommonCrawl tokens  | 50%   | 74%   |  81%  |
|---------------------|-------|-------|-------|
| wiki lempos         | 66.5% | 81.1% | 83.7% |
|---------------------|-------|-------|-------|
| wiki lempos func    | 63.1% | 78.6% | 82.7% |
|---------------------|-------|-------|-------|








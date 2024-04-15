# Algorithmic-Semantic-Primitives
By Aspen Drucker, Ethan Eschrich, and Daniel Soto
## TO-DO
- [ ] Compile dictionaries from Wiktionary
- [ ] Convert to graph + lemmatization
- [ ] Run Genetic Algorithm
- [ ] Compare Word Embeddings

## Dictionaries
As of now the plan is to use Russian, Turkish, and Vietnamese, though Russian can be swapped for pretty much any Indo-European language
This is because they appear to have reasonably large wiktionaries, which means we shouldn't have a poverty of data, and each is included in the MUSE word embeddings and STANZA lemmatizer, so we shouldn't lack tools

See https://github.com/tatuylonen/wiktextract/tree/master for the source of the parser used for Russian

## Convert to graph
Library used for removing stopwords: https://github.com/stopwords-iso/stopwords-iso
Library used for lemmatization: https://stanfordnlp.github.io/stanza/ner_models.html

## Run Genetic Algorithm
Use a subset of the full dictionaries

## Word Embeddings
https://github.com/facebookresearch/MUSE

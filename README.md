# IS469 GenAI with LLMs

Simple notebooks for basic NLP vectorization workflows.

## Contents
- `tfidf_bow_corpus.ipynb`: TF-IDF and Bag-of-Words basics
- `word2vec.ipynb`: Word2Vec training and exploration

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install scipy numpy gensim
```

## Run
```bash
jupyter lab
```

## Notes
- `word2vec.ipynb` downloads the 20-newsgroups dataset at runtime.
- Word2Vec model artifacts are saved locally if you run the save cell.

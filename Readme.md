# Lexical Substitution

This project is a lexical substitution task using:
- part2: WordNet Frequency Baseline
- part3: Simple Lesk Algorithm
- part4: pre-trained Word2Vec embeddings
- part5: BERT masked language model


## installation

1. Install NLTK - Natural Language Toolkit
```python
pip install nltk

python
>>> import nltk
>>> nltk.download()
```
In the Corpora tab, install wordnet and stopwords.

2. Install gensim, a vector space modeling package for Python.
3. Install Huggingface Transformers, BERT implementation by Huggingface (an NLP company), or more specifically their slightly more compact model DistilBERT
```python
pip install gensim
pip install transformers
```

## Usage
```python
python lexsub_main.py lexsub_trial.xml > part5.predict
```

## Output
lexical substitution outputs are stored in .predict files

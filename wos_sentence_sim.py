import pandas as pd
import random
from eunjeon import Mecab
import numpy as np


def word_ngram(bow, num_gram):
    text = tuple(bow)
    ngrams = [text[x:x + num_gram] for x in range(0, len(text))]
    return tuple(ngrams)


def similarity(doc1, doc2):
    cnt = 0
    for token in doc1:
        if token in doc2:
            cnt = cnt+1
    return cnt/len(doc1)


def find_index(num, data):
    index = []
    for i in range(len(data)):
        if data[i] == num:
            index.append(i)
    return index


komoran = Mecab()

text = open('wos_sentence_files', 'r', encoding='utf-8').read().split('\n')
text = np.array(text, dtype='<U200')

text.sort()

bows = []
docs = []

for i in range(len(text)):
    bow2 = komoran.nouns(text[i])
    doc2 = word_ngram(bow2, 2)
    docs.append(doc2)


def makeanswer(query):
    bow1 = komoran.nouns(query)
    doc1 = word_ngram(bow1, 2)

    similar = []

    ignore_idx = text.searchsorted(query)

    for i in range(len(text)):
        doc2 = docs[i]
        try:
            similar.append(similarity(doc1, doc2))
        except:
            similar.append(0)

    similar[ignore_idx] = 0
    idxs = np.array(similar).argsort()

    answer = text[idxs[100]]
    return answer

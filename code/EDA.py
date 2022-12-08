# -*- coding: utf-8 -*-
"""EDA 
TO DO:
eval_df visualizations
"""


"""## Packages"""

import nltk
import string
from spacy.lang.en import English
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

os.getcwd()

train_path = "/content/drive/MyDrive/Colab Notebooks/HuggingFace/dataset/train_df.csv"
eval_path = "/content/drive/MyDrive/Colab Notebooks/HuggingFace/dataset/eval_df.csv"

train_df = pd.read_csv(train_path)
eval_df = pd.read_csv(eval_path)

train_df.head()

train_df['sum_text_ratio'].min(),train_df['sum_text_ratio'].max()

import re
def TextCleaning(text):

    pattern1 = re.compile(r'\<.*?\>')
    s = re.sub(pattern1, '', text)

    pattern2 = re.compile(r'\n')
    s = re.sub(pattern2, ' ', s)

    pattern3 = re.compile(r'[^0-9a-zA-Z!/?]+')
    s = re.sub(pattern3, ' ', s)

    pattern4 = re.compile(r"n't")
    s = re.sub(pattern4, " not", s)

    return s

train_df['text'] = train_df['text'].apply(TextCleaning)

train_df['summary'] = train_df['summary'].apply(TextCleaning)

train_df.shape

train_df['text'][0]

train_df['summary'][0]

train_df['summary'].str.len().hist()
train_df['text'].str.len().hist()
plt.title('Histogram of Text Lenght vs Summary Lenght')
plt.legend(['summary','text'])
plt.show()

train_df['summary'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
train_df['text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
plt.title('Text vs Summary Average Word Lenght')
plt.legend(['summary','text'])
plt.show()

stop= nltk.corpus.stopwords.words('english')

corpus=[]
new = train_df['text'].str.split()
new=new.values.tolist()
corpus_text=[word for i in new for word in i if word not in stop and len(word) > 3]
counter=Counter(corpus_text)
most=counter.most_common()
x, y= [], []
for word,count in most[:10]:
    if (word not in stop):
        x.append(word)
        y.append(count)

sns.barplot(x=y,y=x)
plt.title('Top 10 Word Count in Text')
plt.show()

corpus=[]
new = train_df['summary'].str.split()
new=new.values.tolist()
corpus_summary =[word for i in new for word in i if word not in stop and len(word) > 3]
counter=Counter(corpus_summary)
most=counter.most_common()
x, y= [], []
for word,count in most[:10]:
    if (word not in stop):
        x.append(word)
        y.append(count)

sns.barplot(x=y,y=x)
plt.title('Top 10 Word Count in Summary')
plt.show()

import spacy 
nlp = spacy.load("en_core_web_sm")

def getting_pos(text):
   doc = nlp(text)
   v = 0
   n = 0
   adj = 0
   adv = 0
   num = 0
   for token in doc:
     if token.pos_ == 'VERB':
        v += 1
     elif token.pos_ == 'NOUN':
        n += 1   
     elif token.pos_ == 'ADJ':
       adj += 1
     elif token.pos_ == 'ADV':
       adv += 1     
     elif token.pos_ == 'NUM':
       num += 1  
     else:
        continue 
   return v,n,adj,adv,num

train_df['verbs'], train_df['noun'], train_df['adj'], train_df['adv'], train_df['num'] = zip(*train_df['summary'].apply(getting_pos))

train_df[['verbs','noun','adj','adv','num']].sum().plot.bar()
plt.title('POS of Summary Total Count')
plt.show()

train_df['verbs'], train_df['noun'], train_df['adj'], train_df['adv'], train_df['num'] = zip(*train_df['text'].apply(getting_pos))

train_df[['verbs','noun','adj','adv','num']].sum().plot.bar()
plt.title('POS of Text Total Count')
plt.show()

def plot_top_ngrams_barchart(text, n=2):

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x)

plot_top_ngrams_barchart(train_df['text'],3)
plt.title('Top trigrams in text')
plt.show()

plot_top_ngrams_barchart(train_df['summary'],3)
plt.title('Top trigrams in summary')
plt.show()

# -*- coding: utf-8 -*-
"""EDA

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
import spacy 


os.getcwd()

train_path = "/content/drive/MyDrive/Colab Notebooks/HuggingFace/dataset/train_df.csv"
eval_path = "/content/drive/MyDrive/Colab Notebooks/HuggingFace/dataset/eval_df.csv"

train_df = pd.read_csv(train_path)
eval_df = pd.read_csv(eval_path)

"""## 1 Phase


"""

train_df.head()

train_df['sum_text_ratio'].min(),train_df['sum_text_ratio'].max()

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

#train_df['summary'].str.len().hist()
train_df['text'].str.len().hist()
plt.title('Histogram of Text Lenght')
plt.legend(['summary','text'])
plt.show()

train_df['summary'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
train_df['text'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
plt.title('Text vs Summary Average Word Lenght')
plt.legend(['summary','text'])
plt.show()

train_df['sum_text_ratio'].hist()
plt.title('Histogram of Text Lenght vs Summary Lenght')
plt.show()

sns.histplot(data=train_df, x="sum_text_ratio")
plt.title('Sum text ratio scores histogram')
plt.show()

nltk.download('stopwords')

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

"""
This was the EDa code from our first project which was a bert classfication model 
which we didnt use since we changed projects.

"""
# import nltk
# import string
# from spacy.lang.en import English
# import pandas as pd
# import re
# import os
# import kaggle
# from kaggle.api.kaggle_api_extended import KaggleApi
# import zipfile
# import matplotlib.pyplot as plt


# """ Kaggle Api Download Dataset """
# os.environ['KAGGLE_USERNAME'] = 'koyanjo'
# os.environ['KAGGLE_KEY'] = '33bfba07e0815efc297a1a4488dbe6a3'
# os.chdir(os.getcwd() + '\\data')
# api = KaggleApi()
# api.authenticate()
# api.dataset_download_files('timmayer/covid-news-articles-2020-2022')

# with zipfile.ZipFile('covid-news-articles-2020-2022.zip', 'r') as zipref:
#     zipref.extractall()

# """ EDA """

# df = pd.read_csv('covid_articles_raw.csv')
# print(df['category'].unique())
# lis = ['general', 'business', 'tech', 'science', 'esg']
# df = df[df['category'].isin(lis)]
# print(df['category'].value_counts())
# general = df[df['category'] == 'general'].sample(n = 200, random_state = 1)
# business = df[df['category'] == 'business'].sample(n = 200, random_state = 1)
# tech = df[df['category'] == 'tech'].sample(n = 200, random_state = 1)
# science = df[df['category'] == 'science'].sample(n = 200, random_state = 1)
# esg = df[df['category'] == 'esg'].sample(n = 200, random_state = 1)

# content = general['content'].tolist()
# content_text = ''.join(map(str, content)).replace('\n',' ')
# content_bus = business['content'].tolist()
# content_bus_text = ''.join(map(str, content_bus)).replace('\n',' ')
# content_tech = tech['content'].tolist()
# content_tech_text = ''.join(map(str, content_tech)).replace('\n',' ')
# content_science = science['content'].tolist()
# content_science_text = ''.join(map(str, content_science)).replace('\n',' ')
# content_esg = esg['content'].tolist()
# content_esg_text = ''.join(map(str, content_esg)).replace('\n',' ')

# stopwords = nltk.corpus.stopwords.words('english')
# def common_words(text):
#     allWords = nltk.tokenize.word_tokenize(text)
#     allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.isalpha() and w not in stopwords and len(w) > 3)
#     return allWordExceptStopDist.most_common(10)

# print('general :',common_words(content_science_text))
# print('business :',common_words(content_bus_text))
# print('tech :',common_words(content_tech_text))
# print('science :',common_words(content_science_text))
# print('esg :',common_words(content_esg_text))

# general['title'].str.len().hist()
# business['title'].str.len().hist()
# tech['title'].str.len().hist()
# science['title'].str.len().hist()
# esg['title'].str.len().hist()
# plt.title('Numbers of Characters present in each Headline')
# plt.legend(['general','business','tech','science','esg'])
# plt.show()
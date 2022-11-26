import nltk
import string
from spacy.lang.en import English
import pandas as pd
import re
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import matplotlib.pyplot as plt


""" Kaggle Api Download Dataset """
os.environ['KAGGLE_USERNAME'] = 'koyanjo'
os.environ['KAGGLE_KEY'] = '33bfba07e0815efc297a1a4488dbe6a3'
os.chdir(os.getcwd() + '\\data')
api = KaggleApi()
api.authenticate()
api.dataset_download_files('timmayer/covid-news-articles-2020-2022')

with zipfile.ZipFile('covid-news-articles-2020-2022.zip', 'r') as zipref:
    zipref.extractall()

""" EDA """

df = pd.read_csv('covid_articles_raw.csv')
print(df['category'].unique())
lis = ['general', 'business', 'tech', 'science', 'esg']
df = df[df['category'].isin(lis)]
print(df['category'].value_counts())
general = df[df['category'] == 'general'].sample(n = 200, random_state = 1)
business = df[df['category'] == 'business'].sample(n = 200, random_state = 1)
tech = df[df['category'] == 'tech'].sample(n = 200, random_state = 1)
science = df[df['category'] == 'science'].sample(n = 200, random_state = 1)
esg = df[df['category'] == 'esg'].sample(n = 200, random_state = 1)

content = general['content'].tolist()
content_text = ''.join(map(str, content)).replace('\n',' ')
content_bus = business['content'].tolist()
content_bus_text = ''.join(map(str, content_bus)).replace('\n',' ')
content_tech = tech['content'].tolist()
content_tech_text = ''.join(map(str, content_tech)).replace('\n',' ')
content_science = science['content'].tolist()
content_science_text = ''.join(map(str, content_science)).replace('\n',' ')
content_esg = esg['content'].tolist()
content_esg_text = ''.join(map(str, content_esg)).replace('\n',' ')

stopwords = nltk.corpus.stopwords.words('english')
def common_words(text):
    allWords = nltk.tokenize.word_tokenize(text)
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.isalpha() and w not in stopwords and len(w) > 3)
    return allWordExceptStopDist.most_common(10)

print('general :',common_words(content_science_text))
print('business :',common_words(content_bus_text))
print('tech :',common_words(content_tech_text))
print('science :',common_words(content_science_text))
print('esg :',common_words(content_esg_text))

general['title'].str.len().hist()
business['title'].str.len().hist()
tech['title'].str.len().hist()
science['title'].str.len().hist()
esg['title'].str.len().hist()
plt.title('Numbers of Characters present in each Headline')
plt.legend(['general','business','tech','science','esg'])
plt.show()

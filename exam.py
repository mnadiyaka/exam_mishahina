import numpy as np
import pandas as pd
​
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
​
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
​
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
​
train = pd.read_csv('../input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv', delimiter = "\t", encoding = 'utf-8')
test = pd.read_csv('../input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/testData.tsv', delimiter = "\t", encoding = 'utf-8')
​
sub=pd.read_csv('../input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/sampleSubmission.csv')
train.head()
​
print(train.shape)
print(test.shape)
train['length']=train['review'].apply(len)
train['length'].describe()
​
print(train['review'][0])
​
train_len=train['review'].apply(len)
test_len=test['review'].apply(len)
​
import matplotlib.pyplot as plt
import seaborn as sns
fig=plt.figure(figsize=(15,4))
fig.add_subplot(1,2,1)
sns.distplot((train_len),color='red')
​
fig.add_subplot(1,2,2)
sns.distplot((test_len),color='blue')
​
train['word_n'] = train['review'].apply(lambda x : len(x.split(' ')))
test['word_n'] = test['review'].apply(lambda x : len(x.split(' ')))
​
fig=plt.figure(figsize=(15,4))
fig.add_subplot(1,2,1)
sns.distplot(train['word_n'],color='red')
​
fig.add_subplot(1,2,2)
sns.distplot(test['word_n'],color='blue')
​
train['word_n'].describe()
​
cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis('off')

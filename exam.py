import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

train = pd.read_csv('../input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv', delimiter = "\t", encoding = 'utf-8')
test = pd.read_csv('../input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/testData.tsv', delimiter = "\t", encoding = 'utf-8')

sub=pd.read_csv('../input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/sampleSubmission.csv')
train.head()

print(train.shape)
print(test.shape)
train['length']=train['review'].apply(len)
train['length'].describe()

print(train['review'][0])

train_len=train['review'].apply(len)
test_len=test['review'].apply(len)

import matplotlib.pyplot as plt
import seaborn as sns
fig=plt.figure(figsize=(15,4))
fig.add_subplot(1,2,1)
sns.distplot((train_len),color='red')

fig.add_subplot(1,2,2)
sns.distplot((test_len),color='blue')

train['word_n'] = train['review'].apply(lambda x : len(x.split(' ')))
test['word_n'] = test['review'].apply(lambda x : len(x.split(' ')))

fig=plt.figure(figsize=(15,4))
fig.add_subplot(1,2,1)
sns.distplot(train['word_n'],color='red')

fig.add_subplot(1,2,2)
sns.distplot(test['word_n'],color='blue')

train['word_n'].describe()

cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis('off')

fig, axe = plt.subplots(1,3, figsize=(23,5))
sns.countplot(train['sentiment'], ax=axe[0])
sns.boxenplot(x=train['sentiment'], y=train['length'], data=train, ax=axe[1])
sns.boxenplot(x=train['sentiment'], y=train['word_n'], data=train, ax=axe[2])

stopwords=stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()

def cleaning(raw_text):
    # Removing HTML Tags
    html_removed_text=bs(raw_text).get_text()
    
    # Remove any non character
    character_only_text=re.sub("[^a-zA-Z]"," ",html_removed_text)
    
    # Lowercase and split
    lower_text=character_only_text.lower().split()
    
    # Get STOPWORDS and remove
    stop_remove_text=[i for i in lower_text if not i in stopwords]
    
    # Remove one character words
    lemma_removed_text=[word for word in stop_remove_text if len(word)>1]
    
    #Lemmatization
    lemma_removed_text=[wordnet_lemmatizer.lemmatize(word,'v') for word in stop_remove_text]
    
    
    return " ".join(lemma_removed_text)
cleaning(train['review'][0])

train['cleaned_review']=train['review'].apply(cleaning)
train.head()

test['review']=test['review'].apply(cleaning)

X=train['cleaned_review'] #Predictors
y=train['sentiment'] #Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def create_vector(vectorizer,data):
    '''Pass vectorizer and data'''
    train_vector=vectorizer.transform(data.tolist())
    return train_vector.toarray()
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 20000)
vectorizer.fit(X_train.tolist())

X_train_vector=create_vector(vectorizer,X_train)
X_test_vector=create_vector(vectorizer,X_test)

model_RMC=RandomForestClassifier(n_estimators=110)
model_RMC.fit(X_train_vector,y_train)


y_pred=model_RMC.predict(X_test_vector)
print(classification_report(y_test,y_pred))

test_feature_vector=create_vector(vectorizer,test['review'])
test_predictions=model_RMC.predict(test_feature_vector)

test['sentiment']=test_predictions
test[['id','sentiment']].to_csv("submission.csv",index=False)

print(test["review"])
res=pd.read_csv('./submission.csv')
print(res)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#9.1.1 자연어처리 용어 및 프로세스


# In[ ]:


get_ipython().system('pip install nltk')


# In[ ]:


import nltk
nltk.download("popular")
text=nltk.word_tokenize("Is it possible distinguishing cats and dogs")
text


# In[ ]:


nltk.download('averaged_perceptron_tagger')


# In[ ]:


nltk.pos_tag(text)


# In[ ]:


# 9.1.2 자연어처리를 위한 라이브러리
# NLTK


# In[ ]:


import nltk
nltk.download('punkt')
string1="my favorite subject is math"
string2="my favorite subject is math, english, economic and computer science"
nltk.word_tokenize(string1)


# In[ ]:


nltk.word_tokenize(string2)


# In[ ]:


# KoNLPy

get_ipython().system('pip install konlpy')


# In[ ]:


from konlpy.tag import Komoran
komoran = Komoran()
print(komoran.morphs('딥러닝이 쉽나요? 어렵나요?'))


# In[ ]:


print(komoran.pos('소파 위에 있는 것이 고양이인가요? 강아지인가요?'))


# In[ ]:


#9.2 전처리
#9.2.1 결측치 확인하기


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

import pandas as pd
df = pd.read_csv('class2.csv')
df


# In[ ]:


df.isnull().sum()


# In[ ]:


df.isnull().sum() / len(df)


# In[ ]:


df = df.dropna(how='all')
print(df)


# In[ ]:


df1 = df.dropna()
print(df1)


# In[ ]:


df2=df.fillna(0)
print(df2)


# In[ ]:


df['x'].fillna(df['x'].mean(), inplace=True)
print(df)


# In[ ]:


#9.2.2 토큰화


# In[ ]:


from nltk import sent_tokenize
text_sample = 'Natural Language Processing, or NLP, is the process of extracting the meaning, or intent, behind human language. In the field of Conversational artificial intelligence (AI), NLP allows machines and applications to understand the intent of human language inputs, and then generate appropriate responses, resulting in a natural conversation flow.'
tokenized_sentences = sent_tokenize(text_sample)
print(tokenized_sentences)


# In[ ]:


from nltk import word_tokenize
sentence = " This book is for deep learning learners"
words = word_tokenize(sentence)
print(words)


# In[ ]:


from nltk.tokenize import WordPunctTokenizer  
sentence = "it’s nothing that you don’t already know except most people aren’t aware of how their inner world works."
words = WordPunctTokenizer().tokenize(sentence)
print(words)


# In[ ]:


from tensorflow.keras.preprocessing.text import text_to_word_sequence
sentence = "it’s nothing that you don’t already know except most people aren’t aware of how their inner world works."
words = text_to_word_sequence(sentence)
print(words)


# In[ ]:


#한국어 토큰화 예제


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

import csv
from konlpy.tag import Okt
from gensim.models import word2vec

f = open(r'ratings_train.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
rdw = list(rdr)
f.close()


# In[ ]:


twitter = Okt()

result = []
for line in rdw:
    malist = twitter.pos( line[1], norm=True, stem=True)
    r = []
    for word in malist:
        if not word[1] in ["Josa","Eomi","Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    result.append(rl)
    print(rl)


# In[ ]:


with open("NaverMovie.nlp",'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))


# In[ ]:


mData = word2vec.LineSentence("NaverMovie.nlp")
mModel =word2vec.Word2Vec(mData, size=200, window=10, hs=1, min_count=2, sg=1)
mModel.save("NaverMovie.model")


# In[ ]:


#9.2.3 불용어 제거


# In[ ]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

sample_text = "One of the first things that we ask ourselves is what are the pros and cons of any task we perform."
text_tokens = word_tokenize(sample_text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
print("불용어 제거 미적용:", text_tokens, '\n')
print("불용어 제거 적용:",tokens_without_sw)


# In[ ]:


#9.2.4 어간 추출


# In[ ]:


#포터 알고리즘
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

print(stemmer.stem('obesses'),stemmer.stem('obssesed'))
print(stemmer.stem('standardizes'),stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))
print(stemmer.stem('absentness'), stemmer.stem('absently'))
print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))


# In[ ]:


#랭커스터 알고리즘
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('obsesses'),stemmer.stem('obsessed'))
print(stemmer.stem('standardizes'),stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))
print(stemmer.stem('absentness'), stemmer.stem('absently'))
print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))


# In[ ]:


#표제어 추출(Lemmatization)
import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

print(stemmer.stem('obsesses'),stemmer.stem('obsessed'))
print(lemma.lemmatize('standardizes'),lemma.lemmatize('standardization'))
print(lemma.lemmatize('national'), lemma.lemmatize('nation'))
print(lemma.lemmatize('absentness'), lemma.lemmatize('absently'))
print(lemma.lemmatize('tribalical'), lemma.lemmatize('tribalicalized'))


# In[ ]:


print(lemma.lemmatize('obsesses', 'v'),lemma.lemmatize('obsessed','a'))
print(lemma.lemmatize('standardizes','v'),lemma.lemmatize('standardization','n'))
print(lemma.lemmatize('national','a'), lemma.lemmatize('nation','n'))
print(lemma.lemmatize('absentness','n'), lemma.lemmatize('absently','r'))
print(lemma.lemmatize('tribalical','a'), lemma.lemmatize('tribalicalized','v'))


# In[ ]:


# 9.2.5정규화(Normalization)


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

df = pd.read_csv('covtype.csv')
x = df[df.columns[:54]]
y = df.Cover_Type

x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)
model = tf.keras.Sequential([
 tf.keras.layers.Dense(64, activation='relu',                  
 input_shape=(x_train.shape[1],)),
 tf.keras.layers.Dense(64, activation='relu'),
 tf.keras.layers.Dense(8, activation='softmax')
 ])

model.compile(optimizer= tf.keras.optimizers.Adam(0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

history1 = model.fit(
 x_train, y_train,
 epochs= 26, batch_size = 60,
 validation_data = (x_test, y_test))


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

from sklearn import preprocessing
df = pd.read_csv('covtype.csv')
x = df[df.columns[:55]]
y = df.Cover_Type
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)


train_norm = x_train[x_train.columns[0:10]]
test_norm = x_test[x_test.columns[0:10]]


std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)


training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
x_train.update(training_norm_col)
print (x_train.head())


x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)
print (x_test.head())


# In[ ]:


model = tf.keras.Sequential([
 tf.keras.layers.Dense(64, activation='relu',                  
 input_shape=(x_train.shape[1],)),
 tf.keras.layers.Dense(64, activation='relu'),
 tf.keras.layers.Dense(8, activation=  'softmax')
 ])

model.compile(optimizer= tf.keras.optimizers.Adam(0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
history2 = model.fit(
 x_train, y_train,
 epochs= 26, batch_size = 60,
 validation_data = (x_test, y_test))


# In[ ]:





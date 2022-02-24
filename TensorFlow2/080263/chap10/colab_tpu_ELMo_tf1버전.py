#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip uninstall tensorflow')


# In[2]:


get_ipython().system('pip install tensorflow==1.15')


# In[3]:


get_ipython().system('pip uninstall keras')


# In[4]:


get_ipython().system('pip install keras==2.2.4')


# In[ ]:


#설치후 런타임을 다시 시작해주세요.


# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[2]:


import keras
print(keras.__version__)


# In[3]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
import keras
import numpy as np

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

data = pd.read_csv('spam.csv', encoding='latin-1')

y = list(data['v1'])
x = list(data['v2'])

le = preprocessing.LabelEncoder()
le.fit(y)

def encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)

test = encode(le, ['ham', 'spam', 'ham', 'ham'])

untest = decode(le, test)

x_enc = x
y_enc = encode(le, y)

x_train = np.asarray(x_enc[:5000])
y_train = np.asarray(y_enc[:5000])

x_test = np.asarray(x_enc[5000:])
y_test = np.asarray(y_enc[5000:])

from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K

def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(2, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    history = model.fit(x_train, y_train, epochs=1, batch_size=32)
    model.save_weights('./elmo-model.h5')

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo-model.h5')  
    predicts = model.predict(x_test, batch_size=32)

y_test = decode(le, y_test)
y_preds = decode(le, predicts)

from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_preds))

print(metrics.classification_report(y_test, y_preds))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


#10.1.1 희소표현(Sparse Representation)


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

import pandas as pd
class2=pd.read_csv("class2.csv")

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_x = label_encoder.fit_transform(class2['class2'])
train_x


# In[ ]:


#10.1.2 횟수기반 임베딩
#Counter Vector


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is last chance.',
    'and if you do not have this chance.',
    'you will never get any chance.',
    'will you do get this one?',
    'please, get this chance',
]
vect = CountVectorizer()
vect.fit(corpus)
vect.vocabulary_


# In[ ]:


vect.transform(['you will never get any chance.']).toarray()


# In[ ]:


vect = CountVectorizer(stop_words=["and", "is", "please", "this"]).fit(corpus)
vect.vocabulary_


# In[ ]:


#TF-IDF


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
doc = ['I like machine learning', 'I love deep learning', 'I run everyday']
tfidf_vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
doc_distance = (tfidf_matrix * tfidf_matrix.T)
print ('유사도를 위한', str(doc_distance.get_shape()[0]), 'x', str(doc_distance.get_shape()[1]), '행렬을 만들었습니다.')
print(doc_distance.toarray())


# In[ ]:


#10.1.3 예측기반 임베딩
# Word2Vec


# In[ ]:


get_ipython().system('pip install nltk')


# In[ ]:


import nltk
nltk.download("popular")


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
warnings.filterwarnings(action = 'ignore') 
import gensim 
from gensim.models import Word2Vec 
  
sample = open("peter.txt", "r", encoding='UTF8')
s = sample.read() 
  
f = s.replace("\n", " ")
data = [] 
  
for i in sent_tokenize(f):
    temp = [] 
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp) 

data


# In[ ]:


#CBOW


# In[ ]:


model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5)
print("Cosine similarity between 'peter' " +
                 "'wendy' - CBOW : ", 
      model1.similarity('peter', 'wendy'))


# In[ ]:


print("Cosine similarity between 'peter' " +
                 "'hook' - CBOW : ", 
      model1.similarity('peter', 'hook')) 


# In[ ]:


#Skip-gram


# In[ ]:


model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, sg = 1)
print("Cosine similarity between 'peter' " +
          "wendy' - Skip Gram : ", 
    model2.similarity('peter', 'wendy'))


# In[ ]:


print("Cosine similarity between 'peter' " +
            "hook' - Skip Gram : ", 
      model2.similarity('peter', 'hook')) 


# In[ ]:


#FastText


# In[ ]:


from tqdm import tqdm
from gensim.test.utils import common_texts
from gensim.models import FastText

corpus_fname = 'peter.txt' 
corpus = [sent.strip().split(" ") for sent in tqdm(open(corpus_fname, 'r', encoding='utf-8').readlines())] 
model = FastText(corpus, size=4, window=3, min_count=1, iter=10)


# In[ ]:


sim_score = model.wv.similarity('peter', 'wendy')
print(sim_score)


# In[ ]:


sim_score = model.wv.similarity('peter', 'hook')
print(sim_score)


# In[3]:


#https://fasttext.cc/docs/en/pretrained-vectors.html에서 wiki.ko.vec 파일을 따로 내려받으세요. 
# 내려받은 파일을 불러와 실습합니다.
# 파일 크기가 약 2G 이상이므로 PC에서 파일을 불러오는 시간이 오래 걸리기 때문에 구글 드라이브에 파일을 넣어둔 후 불러오겠습니다.
# PC에서 파일을 불러오려면 아래 코드를 주석 해제 후 실습해주세요.
#from google.colab import files # 데이터 불러오기
#file_uploaded=files.upload()

from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


from __future__ import print_function
from gensim.models import KeyedVectors

model_kr = KeyedVectors.load_word2vec_format('/content/drive/MyDrive/Colab Notebooks/wiki.ko.vec') #구글 드라이브에서 파일 경로 복사 후 붙여넣으세요.

#model_kr = KeyedVectors.load_word2vec_format('wiki.ko.vec')  #구글 드라이브가 아니라 PC에서 파일을 불러왔다면 주석 해제 후 실습


# In[ ]:


find_similar_to = '노력'

for similar_word in model_kr.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))


# In[ ]:


similarities = model_kr.wv.most_similar(positive=['동물', '육식동물'], negative=['사람'])
print(similarities)


# In[ ]:


#10.1.4 횟수/예측기반 임베딩
#GloVe


# In[4]:


# PC에서 파일 불러오는 시간이 오래 걸리기 때문에 구글 드라이브에서 불러옵니다.

#구글 드라이브가 아니라 PC에서 파일을 불러오려면 아래 주석 해제
#from google.colab import files # 데이터 불러오기
#file_uploaded=files.upload()

import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/content/drive/MyDrive/Colab Notebooks/glove.6B.100d.txt')  #구글 드라이브에서 파일 경로 복사 후 붙여넣으세요. 
#glove_file = datapath('glove.6B.100d.txt')                                        # PC에서 불러온다면 주석 해제 후 실습
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)


# In[6]:


model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.most_similar('bill')


# In[7]:


model.most_similar('cherry') 


# In[8]:


model.most_similar(negative=['cherry'])


# In[ ]:


result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))


# In[ ]:


def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]
analogy('australia', 'beer', 'france')


# In[ ]:


analogy('tall', 'tallest', 'long')


# In[ ]:


print(model.doesnt_match("breakfast cereal dinner lunch".split()))


# In[ ]:


#10.2 Transformer attention
#10.2.1 Seq2seq


# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
 
import tensorflow as tf
import os 
import io
import re
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split


# In[ ]:


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
 
def preprocess_sentence(w):
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


# In[ ]:


en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))


# In[ ]:


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
 
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
 
    return zip(*word_pairs)
def max_length(tensor):
    return max(len(t) for t in tensor)
 
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)
 
  tensor = lang_tokenizer.texts_to_sequences(lang)
 
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
 
  return tensor, lang_tokenizer
 
def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
 
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
 
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset('spa.txt', num_examples)
 
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
 
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)


# In[ ]:


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
 
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# In[ ]:


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
 
  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state
 
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
 
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE) 


# In[ ]:


class EDAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(EDAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
 
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
 
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights 
attention_layer = EDAttention(10)


# In[ ]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = EDAttention(self.dec_units)
 
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
 
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


# In[ ]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
 
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


# In[ ]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# In[ ]:


def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables)) 
  return batch_loss


# In[ ]:


EPOCHS = 10
 
for epoch in range(EPOCHS):
  start = time.time()
 
  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
 
  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss
 
    if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)
 
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# In[ ]:


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
 
    sentence = preprocess_sentence(sentence)
 
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
 
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + ' '
        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)
 
    return result, sentence, attention_plot


# In[ ]:


def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


# In[ ]:


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)
 
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
 
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'esta es mi vida.')


# In[ ]:


#10.2.2 Bert
get_ipython().system('pip install bert-for-tf2')
get_ipython().system('pip install sentencepiece')


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
movie_reviews = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/IMDB Dataset.csv") #구글 드라이브에서 파일 경로 복사 후 붙여넣으세요.
movie_reviews.isnull().values.any()
movie_reviews.shape


# In[ ]:


def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

reviews = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    reviews.append(preprocess_text(sen))

print(movie_reviews.columns.values)


# In[ ]:


movie_reviews.sentiment.unique()


# In[ ]:


y = movie_reviews['sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))


# In[ ]:


print(reviews[10])


# In[ ]:


print(y[10])


# In[ ]:


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


# In[ ]:


tokenizer.tokenize("don't be so judgmental")


# In[ ]:


tokenizer.convert_tokens_to_ids(tokenizer.tokenize("don't be so judgmental"))


# In[ ]:


def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))
tokenized_reviews = [tokenize_reviews(review) for review in reviews]


# In[ ]:


import random

reviews_with_len = [[review, y[i], len(review)]
                 for i, review in enumerate(tokenized_reviews)]
random.shuffle(reviews_with_len)
reviews_with_len.sort(key=lambda x: x[2])
sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))
BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
next(iter(batched_dataset))


# In[ ]:


import math

TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)


# In[ ]:


class TEXT_MODEL(tf.keras.Model):
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        self.embedding = tf.keras.layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = tf.keras.layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = tf.keras.layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = tf.keras.layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = tf.keras.layers.GlobalMaxPool1D()
        self.dense_1 = tf.keras.layers.Dense(units=dnn_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = tf.keras.layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = tf.keras.layers.Dense(units=model_output_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) 
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        return model_output


# In[ ]:


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 5


# In[ ]:


text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)


# In[ ]:


if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])

text_model.fit(train_data, epochs=NB_EPOCHS)


# In[ ]:


results = text_model.evaluate(test_data)
print(results)


# In[ ]:


import pandas as pd
import bert
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

train_data = pd.read_csv('train.csv')


# In[ ]:


from google.colab import files # 데이터 불러오기
file_uploaded=files.upload()

test_data = pd.read_csv('test.csv')


# In[ ]:


url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2'
bert_layer = hub.KerasLayer(url, trainable=True)


# In[ ]:


FullTokenizer=bert.bert_tokenization.FullTokenizer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer=FullTokenizer(vocab_file,do_lower_case)


# In[ ]:


def bert_encoder(texts, tokenizer, max_len=512):    
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[ ]:


train_input = bert_encoder(train_data, tokenizer, max_len=160)
train_labels = train_data.target.values


# In[ ]:


def build_model(max_len=512):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), 
                                 dtype=tf.int32, name='positional_ids')
    input_segment_ids = tf.keras.layers.Input(shape=(max_len,), 
                                    dtype=tf.int32, name='segment_ids')
    input_mask = tf.keras.layers.Input(shape=(max_len,), 
                              dtype=tf.int32, name='input_mask')
    pooled_output, sequence_output = bert_layer([input_word_ids, 
                                                 input_mask, 
                                                 input_segment_ids])
    clf_output = sequence_output[:, 0, :]
    output = tf.keras.layers.Dense(1, activation='sigmoid')(clf_output)
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_segment_ids], 
                        outputs=output)
    model.compile(optimizer= RMSprop(lr=2e-6), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


# In[ ]:


model = build_model(max_len=160)
model.summary()


# In[ ]:


train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    batch_size=8
)


# In[ ]:


#10.3 한국어 임베딩


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 

text="""과일 가게에 사과가 많이 진열되어 있다
그녀가 나에게 사과한 후, 우리는 친해졌다
애플은 사과 모양을 로고로 사용한다\n"""


# In[ ]:


tok = Tokenizer()
tok.fit_on_texts([text])

vocSize=len(tok.word_index)+1

seqs = list()
for word in text.split("\n"):
    encoded = tok.texts_to_sequences([word])[0]
    for i in range(1, len(encoded)):
        seq = encoded[:i+1]
        seqs.append(seq)
        
maxLen=max(len(i) for i in seqs)

seqs=pad_sequences(seqs ,maxlen=maxLen, padding="pre") 
seqs 


# In[ ]:


seqs = np.array(seqs)
x = seqs[:,:-1]


# In[ ]:


from tensorflow.keras.utils.np_utils import to_categorical
y = seqs[:, -1]
y = to_categorical(y, num_classes = vocSize)


# In[ ]:


from tensorflow.keras.layers import LSTM 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(vocSize, 10, input_length= maxLen-1, ))         
model.add(LSTM(32))
model.add(Dense(vocSize, activation="softmax"))
model.compile(loss="categorical_crossentropy", metrics = ["accuracy"], optimizer ="adam")
model.fit(x,y,epochs=200)


# In[ ]:


def sentGen(model, tok, word, n):  
    sent = ""
    word2=word
    for _ in range(n):  
        encoded = tok.texts_to_sequences([word])[0] 
        encoded = pad_sequences([encoded], maxlen = 7, padding="pre")
        res = model.predict_classes(encoded)

        for w , i in tok.word_index.items(): 
            if i == res:  
                break 
        word = word + " " + w
        sent = sent + " " + w
    sent = word2 + sent 
    return sent 


# In[ ]:


print(sentGen(model, tok, "과일",2))


# In[ ]:





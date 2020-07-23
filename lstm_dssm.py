#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers as layers
import numpy as np
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops


# In[2]:


def load_vocab(vocab_file):
    word_dict = {}
    with open(vocab_file,encoding='utf-8') as f:
        for idx,word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict


# In[3]:


import random
def preprocess(file,NEG):
    with open(file,encoding='utf-8') as f:
        lines = f.readlines()
    docs = []
    queries = []
    for line in lines:
        line = line.strip()
        docs.append(line.split(' ')[0])
        queries.append(line.split(' ')[1].split('、'))
    result = []
    for i,d in enumerate(queries):
        for query in d:
            neg_docs = random.sample(docs,NEG+1)
            data = [query]
            data.append([docs[i]])
            for neg in neg_docs:
                if neg == docs[i] or len(data[1]) == NEG+1:
                    continue
                else:
                    data[1].append(neg)
            result.append(data)
    return result


# In[4]:


def pad_sequence(data,vocab_file='data/vocab.txt',max_len=15):
    vocab = load_vocab(vocab_file)
    result = []
    for i in data:
        query = tf.keras.preprocessing.sequence.pad_sequences(
            [convert_word2id(i[0],vocab)], maxlen=max_len, dtype='int32', padding='post', truncating='post',
            value=0
        )
        docs = tf.keras.preprocessing.sequence.pad_sequences(
            [convert_word2id(k,vocab) for k in i[1]], maxlen=max_len, dtype='int32', padding='post', truncating='post',
            value=0
        )
#         yield query,docs
        result.append((query,docs.reshape(1,-1)))
    return result


# In[5]:


def convert_word2id(query, vocab_map):
    ids = []
    for w in query:
        if w in vocab_map:
            ids.append(vocab_map[w])
        else:
            ids.append(vocab_map['[UNK]'])
#     while len(ids) < max_seq_len:
#         ids.append(vocab_map['[PAD]'])
    return ids


# In[6]:


def gene():
    for i in result:
        yield i,np.array([0])


# In[7]:


def build_vocab(train_file,vocab_file,val_file = None,test_file = None):
    with open(train_file,encoding='utf-8') as f:
        lines = f.readlines()
    if val_file:
        with open(val_file,encoding='utf-8') as f:
            lines.extend(f.readlines())
    if test_file:
        with open(test_file,encoding='utf-8') as f:
            lines.extend(f.readlines())
    lines = [line.strip('\n') for line in lines]
    vocab = {}
    vocab_list = ['[PAD]','[UNK]']
    for line in lines:
        for word in line:
            if word == ' ' or word == '、':
                continue
            if vocab.get(word) == None:
                vocab[word] = 1
            else:
                vocab[word] += 1
#     print(sorted(vocab.items(), key=lambda item:item[1],reverse=True))
    vocab_list.extend([i[0] for i in sorted(vocab.items(), key=lambda item:item[1],reverse=True)])
    vocab_list = [i+'\n' for i in vocab_list]
    with open(vocab_file,'w',encoding='utf-8') as f:
        f.writelines(vocab_list)
    return len(vocab_list)
#     print(vocab)


# In[8]:


class LSTM_DSSM(tf.keras.Model):
    def __init__(self,emb_dim,lstm_dim,vocal_size,NEG,maxlen,dropout=0.0):
        super(LSTM_DSSM,self).__init__()
        self.word2emb = layers.Embedding(vocal_size,emb_dim,mask_zero=True)
        self.lstm = layers.Bidirectional(layers.RNN((tf.keras.experimental.PeepholeLSTMCell(lstm_dim,
                                dropout=dropout,recurrent_dropout=dropout)),time_major=False),merge_mode='ave')
        self.NEG = NEG
        self.maxlen = maxlen
#         self.softmax = layers.Dense(NEG+1, activation=tf.nn.softmax)
    def call(self,inputs):
#         if training:
        query,docs = inputs[:,:self.maxlen],inputs[:,self.maxlen:]
        query_embedding = self.word2emb(query)
        query_mask = self.word2emb.compute_mask(query)
        query_opt = tf.expand_dims(self.lstm(query_embedding,mask=query_mask),axis=1)
        doc_list = []
        for i in range(self.NEG+1):
            doc_embedding = self.word2emb(docs[:,i:i+self.maxlen])
            doc_mask = self.word2emb.compute_mask(docs[:,i:i+self.maxlen])
            doc_opt = self.lstm(doc_embedding,mask=doc_mask)
            doc_list.append(tf.expand_dims(doc_opt,1))
        doc_merge = tf.keras.backend.concatenate(doc_list,axis=1)
        
        query_norm = tf.tile(tf.math.sqrt(tf.math.reduce_sum(input_tensor=tf.square(query_opt), 
                                                   axis=2, keepdims=False)), [1,self.NEG + 1])
        doc_norm = tf.sqrt(tf.math.reduce_sum(input_tensor=tf.square(doc_merge), axis=2, keepdims=False))
        prod = tf.math.reduce_sum(input_tensor=tf.multiply(tf.tile(query_opt, [1, self.NEG+1, 1]), doc_merge), axis=2, keepdims=False)
        norm_prod = tf.math.multiply(query_norm, doc_norm)
        cos_sim = tf.keras.backend.abs(tf.divide(prod,norm_prod))
        prob = tf.keras.backend.softmax(cos_sim)
        return prob


EMB_DIM = 128
LSTM_DIM = 64
NEG = 30
MAXLEN = 15
EPOCH = 8
BATCH_SIZE = 16
DROPOUT = 0.2
LR = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)


# In[18]:


VOCAB_SIZE = build_vocab('data/train_data_10.txt','data/vocab.txt','data/test_data.txt')
# VOCAB_SIZE = 21128
train_data = preprocess('data/train_data_10.txt',NEG)
train_data_pad = pad_sequence(train_data,max_len = MAXLEN)
train_query = np.concatenate([i[0] for i in train_data_pad],axis=0)
train_docs = np.concatenate([i[1] for i in train_data_pad],axis=0)
train_inputs = np.concatenate([train_query,train_docs],axis=1)


# In[19]:


test_data = preprocess('data/test_data.txt',NEG)
test_data_pad = pad_sequence(test_data,max_len = MAXLEN)
test_query = np.concatenate([i[0] for i in test_data_pad],axis=0)
test_docs = np.concatenate([i[1] for i in test_data_pad],axis=0)
test_inputs = np.concatenate([test_query,test_docs],axis=1)


# In[20]:


def log_loss(y_true,y_pred):
        loss = -tf.math.reduce_sum(tf.math.log(tf.slice(y_pred,[0,0],[-1,1])))
        return loss


# In[23]:


model = LSTM_DSSM(EMB_DIM,LSTM_DIM,VOCAB_SIZE,NEG,MAXLEN,DROPOUT)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])


# In[24]:


model.fit(x=train_inputs,y=np.array([0]*len(train_inputs)),batch_size=BATCH_SIZE,
          epochs=EPOCH,validation_data=(test_inputs,np.array([0]*len(test_inputs))),shuffle=True)


# In[25]:


model.save(f'model/model_{EMB_DIM}_{LSTM_DIM}_{NEG}_{MAXLEN}_0.85')


# In[26]:


def get_embedding(text,model,maxlen=20,vocab_file='data/vocab.txt'):
    vocab = load_vocab(vocab_file)
    result = []
    query = tf.keras.preprocessing.sequence.pad_sequences(
        [convert_word2id(text,vocab)], maxlen=maxlen, dtype='int32', padding='post', truncating='post',
        value=0
    )
    return model.get_layer(index=1)(model.get_layer(index=0)(query)).numpy()


# In[27]:


import pandas as pd
np_list = []
from tqdm import tqdm_notebook
for i in tqdm_notebook(pd.read_excel('data/faq_train_baidutrans_plus.xlsx')['stdq'].values.tolist()):
    np_list.append(get_embedding(i,model,MAXLEN,'data/vocab.txt'))
np.save(f'stdq_{EMB_DIM}_{LSTM_DIM}_0.85.npy',np.concatenate(np_list))




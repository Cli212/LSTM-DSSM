import tensorflow as tf
from tensorflow.keras import layers as layers
import numpy as np
import os
path = '/home/admin/he/lstm_dssm_tf2/'
class Predict:
    def __init__(self,model_path='model/model_128_64_30_15_0.85',vocab_file='data/vocab.txt'):
        self.model = tf.keras.models.load_model(path+model_path)
        self.stdq_emb = np.load(path+'stdq_128_64_0.85.npy')
        def load_vocab(vocab_file):
            word_dict = {}
            with open(path+vocab_file,encoding='utf-8') as f:
                for idx,word in enumerate(f.readlines()):
                    word = word.strip()
                    word_dict[word] = idx
            return word_dict
        self.vocab = load_vocab(vocab_file)

    def convert_word2id(self,query):
        ids = []
        for w in query:
            if w in self.vocab:
                ids.append(self.vocab[w])
            else:
                ids.append(self.vocab['[UNK]'])
    #     while len(ids) < max_seq_len:
    #         ids.append(vocab_map['[PAD]'])
        return ids

    def get_embedding(self,text,maxlen=15):
        query = tf.keras.preprocessing.sequence.pad_sequences([self.convert_word2id(text)], maxlen=maxlen,
                                                              dtype='int32', padding='post', truncating='post',value=0)
        return self.model.get_layer(index=1)(self.model.get_layer(index=0)(query)).numpy()

    def cosine_distance(self,a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        if a.ndim==1:
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
        elif a.ndim==2:
            a_norm = np.linalg.norm(a, axis=1, keepdims=True)
            b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        else:
             raise RuntimeError("array dimensions {} not right".format(a.ndim))
        similarity = abs(np.dot(a, b.T)/(a_norm * b_norm) )
    #     dist = 1. - similiarity
        return similarity

    def predict(self,text):
        with open(path+'data/stdqs.txt',encoding='utf-8') as f:
            stdqs = [i.strip() for i in f.readlines()]
        similarity = []
        text_emb = self.get_embedding(text)[0]
        for i,d in enumerate(stdqs):
            similarity.append(self.cosine_distance(text_emb,self.stdq_emb[i]))
        indexs = sorted(range(len(similarity)), key=lambda k: similarity[k],reverse=True)[:2]
#         for i in indexs:
#             print(stdqs[i],similarity[i])
#         return [stdqs[i] for i in indexs]
        return stdqs[indexs[0]]


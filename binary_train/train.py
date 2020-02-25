# -*- coding:utf-8 -*-
# @Time     :2020/2/25 0025 20:25
# @Author   :LiuHao
# @Site     :
# @File     :train.py

from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
from smart_open import open

file_name = "./corpus/proasmdataset.txt"
train_vec = "proasmdatasetVec.txt.model"


def read_corpus(fname, tokens_only=False):
    with open(fname, encoding="UTF-8") as f:
        for i, line in enumerate(f):
            tokens = simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                yield TaggedDocument(tokens, [i])


train_corpus = list(read_corpus(file_name))
test_corpus = list(read_corpus(file_name, tokens_only=True))


# print(train_corpus[:2])
# print(test_corpus[:2])

def train(ftrain):
    model = Doc2Vec(vector_size=100, window=3, cbow_mean=1, min_count=1)
    model.build_vocab(ftrain)
    model.train(ftrain, total_examples=model.corpus_count, epochs=10)
    model.save(train_vec)
    return model


model_dm = train(train_corpus)

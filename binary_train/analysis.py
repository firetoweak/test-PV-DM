# -*- coding:utf-8 -*-
# @Time     :2020/2/25 0025 22:15
# @Author   :LiuHao
# @Site     :
# @File     :analysis.py

from gensim.models import Doc2Vec

output = "proasmdatasetVec.txt.vector"

model = Doc2Vec.load('proasmdatasetVec.txt.model')

model.wv.save_word2vec_format(output, binary=False)

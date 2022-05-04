import os
import numpy as np
from math import log
"""
describe：bm25算法的升级版，附paper，bm25算法只关注了词频，okaTP在关注词频的基础上，文档相关性分析中考虑了两两关键词的距离，越紧密越重要，即紧密度
基本思想为： 两个搜索关键字在一个文档中出现得越近，这个词对儿的权重就越高，两个词同时出现的定义为在某一文档中距离小于等于5.即中间至多有4个词，最小距离为1
如果一个文档包含至少包含两个查询术语的句子，那么该文档相关的概率必须更大。此外，查询词越近，关联概率越高。

es建立索引：文档：文档中的所有词和在文档中出现的位置，
author:Leah(liuye)

"""
path = os.getcwd()

###BM25
##输入候选文档，和当前query文本，候选文档包含属性，   长度，切词列表【（word,(site1,site2)）】
class BM25(object):
    def __init__(self, docs, query):
        self.docs = docs
        self.doc_num = len(docs)
        self.avgdl = sum([doc.len for doc in docs])/self.doc_num
        self.vob = set(np.array([doc.vob for doc in docs]).reshape(-1))
        self.k1 = 1.5
        self.b = 0.75
        self.query = query###query的list列表

    def IDF(self, word):
        if word not in self.vob:
            return 0.0
        number = 0
        for d in self.docs:
            if word in d.vob:
                number += 1

        idf = log((self.doc_num - number + 0.5)/(number + 0.5))
        return idf

    def R(self, word, doc):
        K = self.k1*(1-self.b+self.b*(doc.len/self.avgdl))
        f = doc.count.get(word,0)/doc.len
        score = f*(self.k1+1)/(f+K)
        return score

    def bm25(self, doc):
        res = 0
        for word in self.query:
            res += self.IDF(word)*self.R(word,doc)
        return res

    def docs_score(self):
        score = {}
        for d in self.docs:
            score[d] = self.bm25(d)
        return dict(sorted(score.items(), key=lambda x : x[1], reverse=True))


    def top_doc_id(self, k = 10):
        id = []
        for k,v in self.docs_score().items():
            id.append(k.id)
        return id

class OkaTP(BM25):
    def __init__(self, docs, query, max_dist = 5):
        super().__init__(docs, query)
        self.max_dist = max_dist

    def sum_tpi(self, doc, word1, word2):
        id1_list = doc.w2n.get(word1, [])
        id2_list = doc.w2n.get(word2, [])
        if not (len(id1_list) and len(id2_list)):
            return 0
        res = 0
        for id1 in id1_list:
            for id2 in id2_list:
                if abs(id1-id2)<=self.max_dist:
                    res += 1/(id1-id2)**2

        return res

    def okatp(self, doc):
        query_len = len(self.query)
        K = self.k1*(1-self.b+self.b*(doc.len/self.avgdl))
        res = 0
        for i in range(query_len):
            for j in range(i+1,query_len):
                tpi_sum = self.sum_tpi(doc, self.query[i], self.query[j])
                wd = (self.k1 + 1)*tpi_sum/(K + tpi_sum)
                res += min(self.IDF(self.query[i]), self.IDF(self.query[j])) * wd

        return res

    def final_score(self,doc):
        return self.bm25(doc)+self.okatp(doc)



    def final_docs_score(self):
        score = {}
        for d in self.docs:
            score[d] = self.final_score(d)
        return dict(sorted(score.items(), key=lambda x : x[1], reverse=True))


    def final_top_doc_id(self, k = 10):
        id = []
        for k,v in self.final_docs_score().items():
            id.append(k.id)
        return id[:min(k,len(id))]





if __name__ == "__main__":
    query = []
    docs = []
    obj = OkaTP(docs,query)
    topid = obj.final_top_doc_id()
    print(topid)



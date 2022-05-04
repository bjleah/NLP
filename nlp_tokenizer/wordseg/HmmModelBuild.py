"""
describe:
#采用微软亚洲研究院的标注语料进行训练，数据介绍见readme
#此语料用来做命名实体识别，识别的实体种类为preson, organization, location,geoplitical
#由于为标注语料，即直接统计概率

author:leah(liuye)

"""
import os
from math import log
from wordseg.common import logger
import re
from collections import defaultdict
import pickle

default_raw_path = os.path.dirname(os.getcwd()) + "\\data\\raw\\msra\\msra_train_bio"
MIN_FLOAT = -3.14e100

class gen_hmm_data(object):
    def __init__(self):
        self.state = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GPE", "I-GPE"]
        self.raw_path = default_raw_path
        self.total = 0
        self.raw_dict = {}
        self.start = defaultdict(int)
        self.emit = {}
        self.trans = {"O": {"O": 0, "B-PER": 0, "B-ORG": 0, "B-LOC": 0, "B-GPE": 0},
                      "B-PER": {"O": 0, "B-PER": 0, "I-PER": 0,"B-ORG": 0, "B-LOC": 0, "B-GPE": 0},
                      "I-PER": {"O": 0, "B-PER": 0, "I-PER": 0,"B-ORG": 0, "B-LOC": 0, "B-GPE": 0},
                      "B-ORG": {"O": 0, "B-ORG":0, "I-ORG": 0, "B-PER": 0, "B-LOC": 0, "B-GPE": 0},
                      "I-ORG": {"O": 0, "B-PER": 0, "I-ORG": 0,"B-ORG": 0, "B-LOC": 0, "B-GPE": 0},
                      "B-LOC": {"O": 0, "B-ORG":0, "I-LOC": 0, "B-PER": 0, "B-LOC": 0, "B-GPE": 0},
                      "I-LOC": {"O": 0, "B-PER": 0, "I-LOC": 0,"B-ORG": 0, "B-LOC": 0, "B-GPE": 0},
                      "B-GPE": {"O": 0, "B-ORG":0, "I-GPE": 0, "B-PER": 0, "B-LOC": 0, "B-GPE": 0},
                      "I-GPE": {"O": 0, "B-PER": 0, "I-GPE": 0,"B-ORG": 0, "B-LOC": 0, "B-GPE": 0}}
        self.load_data()
        self.prob_start()
        self.prob_emit()
        self.prob_trans()

    def load_data(self):
        if not os.path.exists(self.raw_path):
            logger.info("there is no raw file......")
            return
        pre = ""
        with open(self.raw_path, "rb") as f:
            try:
                for no, row in enumerate(f):
                    row = row.strip().decode('utf-8')
                    if len(row) == 0:  # 空行
                        continue
                    if row == str(0):
                        continue
                    word, speech = re.compile("[\s]+").split(row)

                    #初始矩阵
                    # logger.info("starting to initialize the start_prob_matrix......")
                    self.total += 1
                    self.start[speech] += 1

                    #发射矩阵
                    # logger.info("starting to initialize the emit_prob_matrix......")
                    if speech not in self.emit:
                        self.emit[speech] = {word:1}
                    else:
                        if word in self.emit[speech]:
                            self.emit[speech][word] += 1
                        else:
                            self.emit[speech][word] = 1

                    #转移矩阵
                    # logger.info("starting to initialize the trans_prob_matrix......")
                    cur = speech
                    if pre == "":
                        pre = speech
                    else:
                        self.trans[pre][cur] += 1
                        pre = speech


                    if word in self.raw_dict:
                        if speech in self.raw_dict[word]:
                            self.raw_dict[word][speech] = self.raw_dict[word][speech]+1
                        else:
                            self.raw_dict[word][speech] = 1
                    else:
                        self.raw_dict[word] = {speech:1}
            except ValueError:
                raise ValueError('invalid raw dictionary entry in %s at Line %s: %s' % (f.name, no, row))
        f.close()
        return


    def prob_start(self): #初始概率
        for k, v in self.start.items():
            self.start[k] = log(int(v)/self.total)
        #确保所有隐状态key都存在，以方便后面的计算，不存在的置成min_float
        for i in self.state:
            if i not in self.start:
                self.start[i] = MIN_FLOAT
        outpath = os.getcwd()+"\\prob_start.p"
        pickle.dump(self.start, open(outpath, "wb"))

    def prob_emit(self):#发射概率，隐状态->显状态
        total = {}
        for k, v in self.emit.items():
            for w, p in v.items():
                if k not in total:
                    total[k] = p
                else:
                    total[k] += p
        for k, v in self.emit.items():
            for w, p in v.items():
                self.emit[k][w] = log(int(p)/total[k])
        for i in self.state:
            if i not in self.emit:
                self.emit[i] = {}

        outpath = os.getcwd()+"\\prob_emit.p"
        pickle.dump(self.emit, open(outpath, "wb"))

    def prob_trans(self):#转移概率，隐状态->隐状态
        total = {}
        for k, v in self.trans.items():
            for s, c in v.items():
                if k not in total:
                    total[k] = c
                else:
                    total[k] += c

            # if total[k] == 0:
            #     del total[k]
        for k, v in self.trans.items():
            for s, c in v.items():
                if total[k] == 0 or c == 0:
                    self.trans[k][s] = MIN_FLOAT
                else:
                    self.trans[k][s] = log(int(c)/total[k])

        outpath = os.getcwd()+"\\prob_trans.p"
        pickle.dump(self.trans, open(outpath, "wb"))



if __name__ == "__main__":
    obj = gen_hmm_data()
    # obj.load_data()
    print(obj.trans)
    # print(obj.start)

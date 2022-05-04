
"""
describe: HMM 维特比算法对句子进行预测，从而进行实体识别
author : leah(liuye)

"""

import re
import os
import pickle

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"

states = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GPE", "I-GPE"]

PrevStatus = {
    'O': ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GPE", "I-GPE"],
    'B-PER': ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GPE", "I-GPE"],
    'I-PER': ["B-PER", "I-PER"],
    'B-ORG': ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GPE", "I-GPE"],
    "I-ORG": ["B-ORG", "I-ORG"],
    "B-LOC": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GPE", "I-GPE"],
    "I-LOC": ["B-LOC", "I-LOC"],
    "B-GPE": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GPE", "I-GPE"],
    "I-GPE": ["B-GPE", "I-GPE"]
}


def load_model():
    cur_path = os.getcwd()
    start_p = pickle.load(open(cur_path + "\\" + PROB_START_P, "rb"))
    trans_p = pickle.load(open(cur_path + "\\" + PROB_TRANS_P, "rb"))
    emit_p = pickle.load(open(cur_path + "\\" + PROB_EMIT_P, "rb"))
    return start_p, trans_p, emit_p


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    for y in states:  # init隐状态初始概率乘以发射矩阵概率，因为sentence的第一个字没有前一个字，就没有相应的转移概率，所以事先乘以一个初始概率
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)#0代表第一个字，y代表第一个字状态，相应的value为当前位置为该状态的首字符到当前位置的概率
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in states)

    return (prob, path[state]) #返回sentence的隐状态列表，和最好路径的prob


def cut(sentence):
    start_p, trans_p, emit_p = load_model()
    prob, pos_list = viterbi(sentence, states, start_p, trans_p, emit_p)
    # print(pos_list)
    # print(prob,pos_list)
    res = {"ORG": [],"PER": [],"LOC":[], "GPE":[]}
    cur = ""
    site = []
    #确保最后一个char不为B或I
    sentence = sentence + "。"
    pos_list = pos_list + ["O"]

    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos[0] == "B":
            site.append(i)
            next = pos_list[i+1][0]
            if next == "I":
                cur = cur + char
            elif next == "B" or next == "O":
                site.append(i)
                res[pos[-3:]].append((char,tuple(site)))
                site = []
        elif pos[0] == "I":
            next = pos_list[i+1][0]
            if next == "I":
                cur = cur + char
            elif next == "B" or next == "O":
                if len(site) == 0:
                    continue
                site.append(i)
                cur = cur + char
                res[pos[-3:]].append((cur,tuple(site)))
                cur = ""
                site = []

        elif pos[0] == "O":
            continue

    return res

# start_p, trans_p, emit_p = load_model()
# prob, pos_list = viterbi(sentence, states, start_p, trans_p, emit_p)
# print(prob,pos_list)

if __name__ == "__main__":
    sentence = "高举邓小平理论伟大旗帜，回顾一个世纪以来中国人民的奋斗历史"
    sentence = "协商会"
    # sentence = "受到有关方面高度重视"
    res = cut(sentence)
    print(res)
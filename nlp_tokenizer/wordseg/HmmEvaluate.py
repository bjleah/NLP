
"""
describe :
预测测试集，并评估模型效果
命名实体识别的标准化评测定义三个指标，
每一个实体类别的精确率，召回率和F1值，和整体的上述指标

author : leah(liuye)
"""

from wordseg.common import logger
from wordseg.HmmPredict import cut
import re
import os
test_data_path = os.path.dirname(os.getcwd()) + "\\data\\raw\\msra\\msra_test_bio"

entity = ["ORG","PER","LOC", "GPE"]

def load_test_data(test_data_path):
    #return :data = {"ORG": [(word,(ind1,ind2))],"PER": [],"LOC":[], "GPE":[]}
    data = {"ORG": [],"PER": [],"LOC":[], "GPE":[]}
    sentences = ""
    tags = []
    with open(test_data_path, "rb") as f:
        try:
            for no, row in enumerate(f):
                row = row.strip().decode('utf-8')
                if len(row) == 0:  # 空行
                    continue
                if row == str(0):
                    continue
                word, speech = re.compile("[\s]+").split(row)
                sentences = sentences + word
                tags.append(speech)
        except ValueError:
            raise ValueError('invalid raw dictionary entry in %s at Line %s: %s' % (f.name, no, row))

    f.close()

    cur = ""
    site = []
    for ind,tag in enumerate(map(lambda x,y: (x,y),sentences,tags)):
        char = tag[0]
        s = tag[1]
        if s[0] == 'B':
            cur = char
            site.append(ind)
        elif s[0] == "I":
            cur = cur + char
        elif s[0] == "O":
            if cur != "":
                site.append(ind - 1)
                data[tags[ind-1][-3:]].append((cur, tuple(site)))
                cur = ""
                site = []

    return sentences, data

def inter(A,B):#求交集,A,B,list
    res = list(set(A).intersection(set(B)))
    res.sort()
    return res

def union(A,B):#求并集
    res = list(set(A).union(set(B)))
    res.sort()
    return res

def diff(A,B):#求差集，A有，B没有
    res = list(set(A).difference(set(B)))
    res.sort()
    return res



def indicator(pred_ind_list,true_ind_list):
    #P精确率---正确识别的命名实体数/识别出的命名实体数
    #R召回率---正确识别的命名实体数/命名实体总数
    #F1值---2*p*r/p+r
    if len(pred_ind_list) == 0 or len(true_ind_list) == 0:
        return 0,0,0
    P = len(inter(pred_ind_list,true_ind_list))/len(pred_ind_list)
    R = len(inter(pred_ind_list,true_ind_list))/len(true_ind_list)
    F1 = 2*P*R/(P+R)
    return P,R,F1

def extract_ind(dic, key):
    res = []
    for item in dic[key]:
        res.append(item[1])
    return res

def debug(true,pred):
    inter1 = inter(true,pred)
    diffrence = diff(true,pred)
    return inter1,diffrence

def evaluate(test_data_path):
    #分成短句，来进行预测
    sentences, true = load_test_data(test_data_path)
    pred = {'ORG': [], 'PER': [], 'LOC': [], 'GPE': []}
    symbol = [",",";","，","；","。",".","?","？","!","！",":","：","、"]
    if sentences[-1] not in symbol:
        sentences = sentences + "。"

    sentence = ""
    last = 0

    for ind,char in enumerate(sentences):
        if char not in symbol:
            sentence = sentence + char
        else:
            # print(sentence)
            pred_cur = cut(sentence)
            pred_new = {}
            for key,value in pred_cur.items():
                tmp = []
                for t in value:
                    tmp.append((t[0],(t[1][0]+last,t[1][1]+last)))
                pred_new[key] = tmp
            last = ind+1
            for p in pred:
                pred[p] = pred[p] + pred_new[p]
            sentence = ""

    # print(true)
    # print(pred)
    all_t = []
    all_p = []

    for key in entity:
        true_l = extract_ind(true, key)
        pred_l = extract_ind(pred, key)
        inter1,diffrence = debug(true[key], pred[key])
        P,R,F1 = indicator(true_l, pred_l)
        logger.info("the Precision value in key %s is %f " % (key,P*100))
        logger.info("the recall value in key %s is %f " % (key, R*100))
        logger.info("the F1 value in key %s is %f " % (key, F1*100))
        all_t.extend(true_l)
        all_p.extend(pred_l)

    # whole_P,whole_R,whole_F1 = indicator(all_t,all_p)
    # logger.info("the whole Precision value is %f " % whole_P*100)
    # logger.info("the whole recall value is %f " % whole_R*100)
    # logger.info("the whole F1 value is %f " % whole_F1*100)

    return


if __name__ == "__main__":
    # sentences,data = load_test_data(test_data_path)
    # print(data)
    evaluate(test_data_path)
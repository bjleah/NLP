
"""
describe : DAG seg includes get_DAG and caculate
author : leah(liuye)
"""

from wordseg.common import *
import math

class DAG(common_functions):
    def __init__(self):
        #加载词典
        super().__init__()#这里直接执行了加载词典操作

    def get_DAG(self, sentence):
        #根据词表，返回句子中每一个index的可到达的右边界，词表中没有的，右边界为其自身
        #dict = {"我们":5,"中国":4,"中国人":1,"我":0,"中":0}
        #sentence = "我们是中国人"
        #res : {0: [1], 1: [1], 2: [2], 3: [4, 5], 4: [4], 5: [5]}
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.word_freq:
                if self.word_freq[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def calc_DAG(self, sentence, DAG, route):
        #动规求解切词的最佳路径，转移方程为，当前index的最佳路径 = max(当前index的所有右边界遍历*其右边界为开始到句子最后的prob)
        #prob的大小由词频决定 log(word_freq/total) = log(word_freq)-log(total)
        # 动规求解最好路径和prob，当前的prob = max(遍历当前index的所有右边界，当前词*当前词后续的prob)
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = math.log(self.total)  # 所以最后的路径应该是从index=0的时候的右边界开始分词，然后依次分词得到最后的结果
        for idx in range(N - 1, -1, -1):
            route[idx] = max((math.log(self.word_freq.get(sentence[idx:x + 1]) or 1) -logtotal + route[x + 1][0], x) for x in DAG[idx])

    def all_cut(self, sentence):
        dag = self.get_DAG(sentence)
        old_j = -1
        for k, L in dag.items():
            if len(L) == 1 and k > old_j:
                yield sentence[k:L[0] + 1]
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j#下一个index赋给上一个词的右边界的最大值
    def DAG_cut(self, sentence):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc_DAG(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        pre = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if RE_ENG.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
                pre = buf
            # 英文末尾相邻英文不做切分
            elif is_letter(l_word[0]) and len(pre) > 0 and is_letter(pre[-1]):#前一个词为英文，当前词第一个字符为字母，需不切，buf住
                buf += l_word
                pre = buf
                x = y
            elif is_letter(l_word[-1]) and y < N and is_letter(sentence[y]):#前一个词为中文，但当前词的最后一个字符为字母，且下一个字符也为字母，需buf住，不切分
                buf += l_word
                pre = buf
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                pre = ''
                x = y
        if buf:
            yield buf
            buf = ''
    def cut(self, sentence):
        #允许包含在句子中的字符对sentence进行split，遍历其block，不能match的block,skip进行切分match,其余直接返回
        blocks = include_char.split(sentence)
        for b in blocks:
            # print(b,"****")
            if not b:
                continue
            elif is_english_word(b):
                yield b
            elif include_char.match(b):
                for w in self.DAG_cut(sentence):
                    yield w
            else:
                for t in skip_char.split(b):
                    yield t




if __name__ == "__main__":
    sentence = "当我披着失落的外衣慢步走回家，在无任何灯光的大街上。我仿佛看到死神在向我招手"
    dag = DAG()
    res = []
    for word in dag.cut(sentence):
        res.append(word)
    print(res)
    # print(len(dag.word_freq))
    # print(dag.word_freq[""])
    # print(dag.word_freq)


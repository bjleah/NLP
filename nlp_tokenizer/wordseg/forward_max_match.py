"""
describe : AC自动机加载词典，并进行前向最大匹配，有些粗糙，但是较快
author : leah(liuye)

"""
from wordseg.common import *
from wordseg.AC import *

class forward(common_functions):
    def __init__(self):
        super().__init__()
        self.ac = Ahocorasick()
        self.AC_load_dict()
    def AC_load_dict(self):
        for word, freq in self.word_freq.items():
            if freq != 0:
                self.ac.addWord(word)
        self.ac.make()
        return
    def cut(self,sentence):
        ac_res = self.ac.search(sentence)
        ac_res.sort()
        res = []#[(word,(site1,site2))]
        cur_s = ac_res[0][0]
        cur_e = ac_res[0][1]
        for s,e in ac_res:
            if s == cur_s:
                if e>cur_e:
                    cur_e = e
            else:
                res.append((sentence[cur_s:cur_e+1],(cur_s,cur_e)))
                if s>cur_e:
                    cur_s = s
                    cur_e = e
        return res


if __name__ == "__main__":
    f = forward()
    sentence = "西边的太阳渐渐落去，一缕阳光仍然没力的穿过那厚厚云层，" \
               "穿过那棵高高的杨桃树，照射到坐在门口老人身上，" \
               "老人脸上显得是那样的无奈。此刻，他双手托着下巴，" \
               "眼里流露着孤独无助的茫然神色。在他的心目中，" \
               "黄昏岁月还能坚持多长呢？"
    res = f.cut(sentence)
    print(res)







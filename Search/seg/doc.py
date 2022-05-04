from collections import Counter
import jieba
"""
author：leah
doc:[(word,(site1,site2),词性)]
将doc封装成类返回
"""
class text(object):
    def __init__(self, doc, id):
        self.len = len(doc)
        self.count = Counter([a[0] for a in doc])
        self.vob = list(self.count.keys())
        self.site = {i[0]:i[1] for i in doc}
        self.id = id
        self.w2n = {}
        self.n2w = {}
        for no, word in enumerate(sorted(doc, key=lambda x:x[1])):
            if word[0] not in self.w2n:
                self.w2n[word[0]] = [no]
            else:
                self.w2n[word[0]].append(no)
            self.n2w[no] = word[0]



if __name__ == "__main__":
    raw = "在秋风萧瑟声中，小小的叶片终于鼓足了勇气，"
    word_list = [('在',(0,0),"v"), ('秋风',(1,2),"n"), ('萧瑟',(3,4),"adj"), ('声中',(5,6),"n"), ('，',(7,7),"m"), ('小小的',(8,10),"adj"),
                 ('叶片',(11,12),"n"), ('终于',(13,14),"adj"), ('鼓足',(15,16),"v"), ('了',(17,17),"m"), ('勇气',(18,19),"n"), ('，',(20,20),"m")]
    # print(word_list)
    doc =text(word_list,id = 1)
    # print(doc.len,doc.count,doc.vob,doc.site)
    print(doc.w2n)
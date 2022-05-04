import re
import os
from log import get_logger
logger = get_logger()

rmrb_file = os.getcwd() + "\\data\\rmrb199801.txt"
rmrb_write_to = os.getcwd() + "\\data\\rmrb.txt"


re_word_match = re.compile("[\s]+([\u0021-\u007e\uff21-\uff3b\uff41-\uff5b\uff10-\uff1a\u4e00-\u9fa5\u2160-\u2180\u03b1-\u03c9\xb0\u3001\u3002]+/[a-zA-Z]+)")
re_word_speech = re.compile("([\u0021-\u007e\uff21-\uff3b\uff41-\uff5b\uff10-\uff1a\u4e00-\u9fa5\u2160-\u2180\u03b1-\u03c9\xb0\u3001\u3002]+)/([a-zA-Z]+)")

"""
describe:process raw data in folder of data ,return data as a list
author:Leah(liuye)
"""

class corpus_p(object):
    def __init__(self):
        self.corpus = ""
        self.tags = ""
        self.pos = []
    def gen_rmrb_data(self):##读取rmrb raw数据
        cur_file = open(rmrb_file, "r", encoding="utf-8")
        for line in cur_file.readlines():
            line = re_word_match.split(line)
            # print(len(line))
            for p in line:
                if re_word_speech.match(p):
                    word, speech = re_word_speech.match(p).groups()
                    self.corpus = self.corpus + word
                    if len(word) > 1:
                        for c in range(len(word)):
                            if c == 0:
                                self.tags += "b"
                                self.pos.append(speech + "_" + "b")
                            elif c == len(word)-1:
                                self.tags += "e"
                                self.pos.append(speech + "_" + "e")
                            else:
                                self.tags += "m"
                                self.pos.append(speech + "_" + "m")

                    elif len(word) == 1:
                        self.tags += "s"
                        self.pos.append(speech + "_" + "s")
                else:
                    continue
        cur_file.close()

        print("assert corpus length equal tags length:",len(self.corpus)==len(self.tags))

        with open(rmrb_write_to, "wb") as fw:
            for char, tag, p in zip(self.corpus, self.tags, self.pos):
                row = str(char + "\t" + tag + "\t" + p + "\n").encode("utf-8")
                fw.write(row)

        fw.close()
    @staticmethod
    def read_rmrb_data(path):#读取人民日报切词的标注数据
        try:
            f = open(path, "r", encoding="utf-8")
        except Exception:
            logger.error("fail to open file......")
            return
        corpus = ""
        tags = ""
        for no, row in enumerate(f, 1):
            try:
                row = row.strip()
                if len(row) == 0:#空行
                    continue
                word, tag = row.split('\t')[:2]
                corpus += word
                tags += tag
            except ValueError:
                raise ValueError('invalid dictionary entry in %s at Line %s: %s' % (f.name, no, row))
        f.close()
        return corpus,tags
    @staticmethod
    def read_rmrb_file(path):#读取人民日报纯文本文件
        path_l = []
        if os.path.isdir(path):
            for f in os.listdir(path):
                path_l.append(os.path.join(path, f))
        else:
            path_l = [path]

        corpus = ""
        for path in path_l:
            logger.info("processing file is :"+path)
            try:
                f = open(path, "r", encoding="utf-8")
            except Exception:
                logger.error("fail to open file......")
                return
            for no, row in enumerate(f, 1):
                try:
                    row = row.strip()
                    if len(row) == 0:  # 空行
                        continue
                    corpus += row
                    if len(corpus) > 100000:
                        yield corpus
                        corpus = ""
                except ValueError:
                    raise ValueError('invalid dictionary entry in %s at Line %s: %s' % (f.name, no, row))
            f.close()
        if len(corpus):
            yield corpus




if __name__ == "__main__":
    # path = os.getcwd() + "\\data\\rmrb_test.txt"
    # corpus,tags = corpus_p.read_rmrb_data(path)
    # print(corpus[:50])
    # print(tags[:50])
    from collections import Counter
    vob_size = Counter()
    path = os.getcwd() + "\\data\\rmrb_file"
    for i in corpus_p.read_rmrb_file(path):
        vob_size.update(i)

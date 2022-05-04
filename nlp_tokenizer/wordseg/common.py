"""
describe : common function definition
author : leah(liuye)
"""


import re
from collections import defaultdict
import logging
import sys
import os

sim_chars = '\u4E00-\u9FD5a'# 简体中文
greek_chars = '\u0391\u0392\u0393\u0394-\u03A9\u03B1-\u03C9'# 希腊字符
other_chars = '\u00B0\u0060\u3001'# 其它特殊字符[°,`、]
alphabet_chars = "a-zA-Z0-9+/#'&\._ -"# 英文字母表字符
number_chars = '\u2160-\u216F\u2170-\u217F'# 阿拉伯数字字符
valid_chars = sim_chars + alphabet_chars + greek_chars + number_chars + other_chars
include_char = re.compile("([%s]+)" % valid_chars)
skip_char = re.compile("(\r\n|\s)")
RE_ENG = re.compile('[a-zA-Z0-9]')

#logger设定，输出到sys.stdout（也可设定输出到日志文件或者循环日志文件）
log_console = logging.StreamHandler(sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_console)

#词典文件原始数据路径
path = os.path.dirname(os.getcwd())+"\\data\\raw\\THUOCL-master\\data"
default_DAG_file_path = [os.path.join(path,file) for file in os.listdir(path)]#list
default_user_file_path = os.path.dirname(os.getcwd())+"\\data\\userdict.txt"
default_freq = 1000000

def is_letter(uchar):
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    return False

def is_english_word(sentence):
    return all(is_letter(char) for char in sentence)

class common_functions(object):

    def __init__(self):
        self.word_freq = {}
        self.total = 0
        self.user_words = defaultdict(int)
        self.DAG_Data_file = default_DAG_file_path
        self.user_dict = default_user_file_path
        for p in self.DAG_Data_file:
            self.gen_DAG_dict(p)
        self.load_user_dict()
    def gen_DAG_dict(self, path):
        logger.debug("trying to build DAG dict......")
        try:
            f = open(path, "r", encoding="utf-8")
        except Exception:
            logger.error("fail to open DAG file......")
            return

        for no, row in enumerate(f, 1):
            try:
                row = row.strip()
                if len(row) == 0:#空行
                    continue
                word, freq = row.split('\t')[:2]
                self.word_freq[str(word)] = int(freq) + 1 #词典中有些词词频为0，加一
                self.total += int(freq)

                #在词典中加入所有词的前缀，为了后续DAG的计算，freq设置为0
                for p in range(len(word)):
                    if not self.word_freq.get(word[:p+1]):
                        self.word_freq[word[:p+1]] = 0
            except ValueError:
                raise ValueError('invalid dictionary entry in %s at Line %s: %s' % (f.name, no, row))
        f.close()
        return


    def load_user_dict(self):
        if not os.path.exists(self.user_dict):
            logger.info("there is no user dict......")
            return
        with open(self.user_dict, "rb") as f:
            try:
                for no, row in enumerate(f):
                    row = row.strip().decode('utf-8')
                    row_l = re.compile("[\s]+").split(row)
                    if len(row_l) == 1:#用户词典只有word
                        self.word_add(row_l[0])
                    elif len(row_l) == 2:#用户词典只有word和freq
                        self.word_add(row_l[0],row_l[1])
                    else:
                        logger.info("%s :%s line is more than two parts...."% (f.name, no))
            except ValueError:
                raise ValueError('invalid user dictionary entry in %s at Line %s: %s' % (f.name, no, row))
        f.close()
        return

    def word_add(self, word, freq = default_freq):
        freq = int(freq)
        word = str(word)
        self.word_freq[word] = freq
        self.total += freq
        self.user_words[word] += freq

        for i in range(len(word)):
            if word[:i+1] not in self.word_freq:
                self.word_freq[word[:i+1]] = 0

    def word_delete(self, word):
        self.word_add(word,0)

if __name__ == "__main__":
    fn = common_functions()
    # fn.gen_DAG_dict()
    # fn.load_user_dict()
    print(fn.user_words)
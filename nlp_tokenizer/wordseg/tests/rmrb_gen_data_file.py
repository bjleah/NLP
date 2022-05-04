import re
import os

rmrb_file = os.path.dirname(os.path.dirname(os.getcwd())) + "\\data\\raw\\rmrb199801.txt"
rmrb_write_to = os.path.dirname(os.path.dirname(os.getcwd())) + "\\data\\raw\\rmrb.txt"
cur_file = open(rmrb_file, "r", encoding = "utf-8")

re_word_match = re.compile("[\s]+([\u4E00-\u9FD5a]+/[a-zA-Z]+)")
re_word_speech = re.compile("([\u4E00-\u9FD5a]+)/([a-zA-Z]+)")

word_dict = {}

for line in cur_file.readlines():
    line = re_word_match.split(line)
    for p in line:
        if re_word_speech.match(p):
            word, speech = re_word_speech.match(p).groups()
            if word not in word_dict:
                word_dict[word] = [1, speech]
            else:
                word_dict[word] = [word_dict[word][0]+1, speech]
        else:
            continue
cur_file.close()

with open(rmrb_write_to, "wb") as fw:
    for k,v in word_dict.items():
        row = str(k + "\t" + str(v[0]) + "\t" + str(v[1]) + "\n").encode("utf-8")
        fw.write(row)

fw.close()


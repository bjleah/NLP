from query_parsing.functions import *
from log import get_logger
logger = get_logger()

class preprocess(object):
    def __init__(self):
        pass

    @staticmethod
    def clean(sentence, log = False):
        sentence = quan2ban(sentence)
        if log:
            logger.info(sentence)
        sentence = strip_punc(sentence)
        if log:
            logger.info(sentence)
        sentence = whitespaces2one(sentence)
        if log:
            logger.info(sentence)
        sentence = fan2s(sentence)
        if log:
            logger.info(sentence)
        sentence = remove_invalid_ch(sentence)
        if log:
            logger.info(sentence)
        sentence = common_ch_style(sentence)
        if log:
            logger.info(sentence)
        return sen_trunc(sentence)



if __name__ == "__main__":
    text = "空山新雨後,天氣晚來秋。明月鬆間照,清泉石上流。 竹喧歸浣女,蓮動下漁舟。隨意春芳歇,王孫自可留。￥%……&*（！@￥λ()"
    res = preprocess.clean(text)
    print(res)
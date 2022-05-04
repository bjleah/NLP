import re
from pyhanlp import HanLP



def quan2ban(string):
    rst = ''
    for ch in string:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        elif code == 0x3000:  # 空格
            code = 0x20
        rst = rst + chr(code)
    return rst

def strip_punc(s, keep_chars=None):
    """
    去除两端标点符号(中英文)，以及空白符
    """
    punctuation = '!"#$%&\'()*+,-./:;<=>?？@[\\]^_`{|}~，。；【】（）￥……—\t\n\x0b\x0c\r '
    rm_set = set(ch for ch in punctuation)
    if keep_chars is not None:
        for ch in keep_chars:
            rm_set.remove(ch)
    rm_chars = ''.join(rm_set)
    rst = s.strip(rm_chars)
    return rst

def common_ch_style(string):
    """转换为字符常用书写方式
    英文字符:大写
    罗马数字:大写
    希腊字符:小写
    """
    rst = ''
    for char in string:
        if '\u0370' <= char <= '\u03ff':#希腊文
            rst += char.lower()
        else:
            rst += char.upper()
    return rst


def remove_invalid_ch(s):
    """
    desc 去除英文字母、符号、数字(包括罗马)汉字之外的字符
    基本符号：\u0020-\u007e
    字母数字的全型形式：\uff21-\uff3b,\uff41-\uff5b,\uff10-\uff1a
    中文：\u4e00-\u9fa5
    中文符号 、:\u3001
    中文符号 。:\u3002
    罗马数字: \u2160-\u2180
    希腊字符: \u0370-\u03ff
    其他特殊字符:°:\xb0
    """

    pt = re.compile('[^\u0020-\u007e\uff21-\uff3b\uff41-\uff5b\uff10-\uff1a\u4e00-\u9fa5\u2160-\u2180\u03b1-\u03c9\xb0\u3001\u3002]')
    rst = re.sub(pt, '', s)
    return rst

def whitespaces2one(s):
    return re.sub('\s+', ' ', s).strip()


#台湾繁体转简体
def fan2s(text):
    return HanLP.tw2s(text)


#上述对query使用


#query截断（避免过长，根据模型的承受能力来处理）
def sen_trunc(sentence,length = 300):
    sentence = sentence[:length]
    return sentence





if __name__ == "__main__":
    text = "，我是。？"
    res = quan2ban(text)
    for i,j in zip(text,res):
        print(len(i.encode()),len(j.encode()))
    print(res)

    text = "?.....,,***锄禾日当午？？？？？？"
    res = strip_punc(text)
    print(res)

    text = "λΠ的水分jis"
    res = common_ch_style(text)
    print(res)


    text = "\、sjigsisGDISHGDUI计算估计￥%……&*（！@￥λ()"
    text = quan2ban(text)
    print(text)
    res = remove_invalid_ch(text)
    print(res)


    text = "空山新雨後,天氣晚來秋。明月鬆間照,清泉石上流。 竹喧歸浣女,蓮動下漁舟。隨意春芳歇,王孫自可留。￥%……&*（！@￥λ()"
    res = fan2s(text)
    print(res)


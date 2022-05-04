"""
此模块用来爬取相关query的百度搜索结果，用于建立简单的es数据库
在具体使用前，请先安装如下链接的库 https://github.com/BaiduSpider/BaiduSpider
author:leah

别用这个库了，实践证明很难用，抓不到几个数据😂
"""
from baiduspider import BaiduSpider
import re
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from log import get_logger
logger = get_logger()

###爬取query
"""
query采用知乎上的经典问题，共分为90多个大类别，包括文学、考古、历史、法律等等，每个大类别会有不同的提问，爬取相关问题
"""
def get_category():
    link = "https://zhuanlan.zhihu.com/p/25964484"
    headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0'}
    html = requests.get(link, headers = headers).text
    soup = BeautifulSoup(html, "lxml")
    content = [str(t) for t in soup.findAll("a", attrs={"internal"})]
    topic_link = {}
    re_href = re.compile("(href=\")(.+)(\")")
    re_cate = re.compile("[0-9]+、(.+)<")
    for no, c in enumerate(content):
        try:
            cate = re_cate.search(c).groups()[0]
            href = re_href.search(c).groups()[1]
            topic_link[cate] = href
        except AttributeError:
            print("no matching data in line %d"%no)
            continue
    return topic_link

def get_query():
    topic_link = get_category()
    cate2q = defaultdict(list)
    f = open(os.getcwd() + "\\data\\query.txt", "wb")
    for k,v in topic_link.items():
        headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0'}
        html = requests.get(v, headers=headers).text
        soup = BeautifulSoup(html, "lxml")
        content = [str(t) for t in soup.findAll("a", attrs={"internal"})]
        re_query = re.compile(">(.+)<")
        logger.info("getting question on %s topics"%k)
        for q in content:
            tmp = re_query.search(q)
            if tmp:
                qu = tmp.groups()[0]
                cate2q[k].append(qu)
                f.write(str(qu + "\n").encode("utf-8"))
    all_q = [qu for v_l in cate2q.values() for qu in v_l]
    f.close()
    return all_q, cate2q

def load_query():
    path = os.getcwd() + "\\data\\query.txt"
    f = open(path, "r", encoding="utf-8")
    question = []
    for no,row in enumerate(f, 1):
        try:
            row = row.strip()
            question.append(row)
        except:
            logger.error("invaild row in f.name at line %d"%no)
    return question

def get_query_related():
    questions = load_query()
    spider = BaiduSpider()
    sen_pair = pd.DataFrame(columns = ["ori", "rel"])
    f = open(os.getcwd()+"\\data\\pair.txt", "wb")
    ind = 0
    for no, q in enumerate(questions):
        obj = spider.search_web(q)
        tmp = obj.related
        print(tmp)
        for r in tmp:
            # sen_pair.loc[ind] = [q,r]
            f.write(str(q + "\t" + r + "\n").encode("utf-8"))
            ind += 1
    f.close()
    # sen_pair.to_csv(os.getcwd()+"\\data\\sen_pair.csv",encoding="utf-8-sig",index = False)
    return
def load_sen_pair():
    df = pd.read_csv(os.getcwd()+"\\data\\sen_pair.csv")
    return df.values


def get_query_search_res():
    pass


if __name__ == "__main__":
    # res = get_category()
    # all_q, cate2q = get_query()
    # print(all_q)
    get_query_related()

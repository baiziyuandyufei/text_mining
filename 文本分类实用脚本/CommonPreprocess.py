#coding:utf-8
"""
通用预处理
"""
from pyhanlp import *
from nltk import ngrams

# 加载实词分词器 参考https://github.com/hankcs/pyhanlp/blob/master/tests/demos/demo_notional_tokenizer.py
Term = JClass("com.hankcs.hanlp.seg.common.Term")
NotionalTokenizer = JClass("com.hankcs.hanlp.tokenizer.NotionalTokenizer")


# 通用预处理（训练语料和预测语料通用）
def preprocess(text):
    # 全部字母转小写
    text =text.lower()
    word_li = []

    #  NotionalTokenizer.segment中有去除停用词的操作
    for term in NotionalTokenizer.segment(text):
        word = str(term.word)
        pos = str(term.nature)
        # 去掉时间词
        if pos == u't':
            continue
        # 去掉单字词（这样的词的出现有可能是因为分词系统未登录词导致的）
        if len(word) == 1:
            continue
        word_li.append(word)

    return word_li


# 通用预处理（训练语料和预测语料通用）保留单字词
def preprocess_single_word(text):
    # 全部字母转小写
    text =text.lower()
    word_li = []

    #  NotionalTokenizer.segment中有去除停用词的操作
    for term in NotionalTokenizer.segment(text):
        word = str(term.word)
        pos = str(term.nature)
        # 去掉时间词
        if pos == u't':
            continue
        word_li.append(word)

    return word_li
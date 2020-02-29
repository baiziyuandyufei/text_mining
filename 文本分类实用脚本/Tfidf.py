#coding:utf-8
"""
关键词抽取tf-idf法
用法：python 类目关键词抽取tf.py 文件名 每个类目最大关键词数量
要求：python3，sklearn，PyHanLP
说明：输入文件中每一行存储一个类目的所有文本。
程序会统计每个词项的tf-idf值，这里的idf指的逆类目频率，
并输出每个类目的按tf-idf值降序的topx个词语，x由第2个参数决定默认为10
"""

import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from CommonPreprocess import preprocess
import sys


def extract_keywords(text_li, topx=10):
    """
    用tf-idf法抽取每个类目的关键词
    :param text_li: 类目文本类表，每个元素表示一个类目的所有文本串
    :param topx: 每个类目抽取出的关键词数量
    :return: 返回每个类目的关键词序列
    """
    tv = TfidfVectorizer(min_df=10, max_df=0.4, analyzer=preprocess)
    tv_fit = tv.fit_transform(text_li)
    print("文本集向量初始化完毕\n样本数量：%d, 词汇表长度(特征数量)：%d" % tv_fit.shape)
    vsm = tv_fit.toarray()
    category_keywords_li = []
    for i in range(vsm.shape[0]):
        sorted_keyword = sorted(zip(tv.get_feature_names(), vsm[i]), key=lambda x:x[1], reverse=True)
        category_keywords = [w[0] for w in sorted_keyword[:topx]]
        category_keywords_li.append(category_keywords)
    return category_keywords_li


def main():
    input_file_name = sys.argv[1]
    # 关键词数量
    topx = int(sys.argv[2])
    # 读入文本抽取关键词
    with codecs.open(input_file_name, 'rb', 'utf-8', 'igonre') as infile:
        text_li = infile.readlines()
    category_keywords_li = extract_keywords(text_li, topx)
    print(category_keywords_li)


if __name__ == "__main__":
    main()

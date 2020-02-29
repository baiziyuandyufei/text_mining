#coding:utf-8
"""
single-pass增量聚类演示
"""

import numpy as np
from sklearn.datasets import load_files
from pyhanlp import *
import re
import codecs

NotionalTokenizer = JClass("com.hankcs.hanlp.tokenizer.NotionalTokenizer")
# 以文本在文本集中的顺序列出的文本向量矩阵（用300维向量表示）
text_vec = None
# 以文本在文本集中的顺序列出的话题序号列表
topic_serial = None
# 话题数量
topic_cnt = 0


# 加载词语向量词典
word_dict = dict()
with codecs.open('../dictionary/cc.zh.300.vec', 'rb', 'utf-8', 'ignore') as infile:
    infile.readline()
    for line in infile:
        line = line.strip()
        if line:
            items_li = line.split()
            word = items_li[0]
            word_vec = np.array([float(w) for w in items_li[1::]])
            word_dict[word] = word_vec
print("load cc.zh.300.vec len = %d" % len(word_dict))


# 仅保留中文字符
def translate(text):
    p2 = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    zh = " ".join(p2.split(text)).strip()
    zh = ",".join(zh.split())
    res_str = zh  # 经过相关处理后得到中文的文本
    return res_str


# 预处理，实词分词器分词，查询词语向量，并返回文本向量
def preprocess(text):
    sen_vec = np.zeros((1, 300))
    # 去掉非中文字符
    text = translate(text)
    # 将\r\n替换为空格
    text = re.sub(u'[\r\n]+', u' ', text)
    # 分词与词性标注，使用实词分词器
    word_li = NotionalTokenizer.segment(text)
    word_li = [w.word for w in word_li]
    # 去掉单字词
    word_li = [w for w in word_li if len(w)>1]
    # 查询每个词语的fasttext向量，计算句子向量
    valid_word_cnt = 0
    for word in word_li:
        if word in word_dict:
            sen_vec += word_dict[word]
            valid_word_cnt += 1
    if valid_word_cnt > 0:
        sen_vec = sen_vec*(1.0/valid_word_cnt)
    # 单位化句子向量
    sen_vec = sen_vec*(1.0/np.linalg.norm(sen_vec))
    return text, sen_vec


# single-pass
def single_pass(sen_vec, sim_threshold):
    global text_vec
    global topic_serial
    global topic_cnt
    if topic_cnt == 0:  # 第1次送入的文本
        # 添加文本向量
        text_vec = sen_vec
        # 话题数量+1
        topic_cnt += 1
        # 分配话题编号，话题编号从1开始
        topic_serial = [topic_cnt]
    else:  # 第2次及之后送入的文本
        # 文本逐一与已有的话题中的各文本进行相似度计算
        sim_vec = np.dot(sen_vec, text_vec.T)
        # 获取最大相似度值
        max_value = np.max(sim_vec)
        # 获取最大相似度值的文本所对应的话题编号
        topic_ser = topic_serial[np.argmax(sim_vec)]
        print("topic_ser", topic_ser, "max_value", max_value)
        # 添加文本向量
        text_vec = np.vstack([text_vec, sen_vec])
        # 分配话题编号
        if max_value >= sim_threshold:
            # 将文本聚合到该最大相似度的话题中
            topic_serial.append(topic_ser)
        else:
            # 否则新建话题，将文本聚合到该话题中
            # 话题数量+1
            topic_cnt += 1
            # 将新增的话题编号（也就是增加话题后的话题数量）分配给当前文本
            topic_serial.append(topic_cnt)


def main():
    # 加载数据
    data_all = load_files(container_path=r'../data/news', categories=u'Sports',
                            encoding=u'gbk', decode_error=u'ignore')
    # 获取文本数据集
    data = data_all.data
    # 预处理后的文本数据集
    preprocessed_data = []
    # 进行增量聚类
    for text in data:
        text, text_vec = preprocess(text)
        single_pass(text_vec, 0.9)
        preprocessed_data.append(text)
    # 输出聚类结果
    with open('res_single_pass.txt', 'wb') as outfile:
        sorted_text = sorted(zip(topic_serial, preprocessed_data), key=lambda x:x[0])
        for topic_ser, text in sorted_text:
            out_str = u'%d\t%s\n' % (topic_ser, text)
            outfile.write(out_str.encode('utf-8', 'ignore'))
    print("program finished")
    # 在mac下释放向量内存时间较长，可以直接ctrl+c强制退出程序


if __name__ == '__main__':
    main()
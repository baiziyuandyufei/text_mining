#coding:utf-8
"""
互信息、卡方、频率征选择示例
"""

import codecs
import re
from math import log2
from sklearn.datasets import load_files
from pyhanlp import *

# 加载实词分词器 参考https://github.com/hankcs/pyhanlp/blob/master/tests/demos/demo_notional_tokenizer.py
Term =JClass("com.hankcs.hanlp.seg.common.Term")
NotionalTokenizer = JClass("com.hankcs.hanlp.tokenizer.NotionalTokenizer")


def getDocuments(root_path, file_path_li, file_encoding="utf-8"):
    """
    读取原始文档集并进行预处理
    :param file_path: 文档集所在路径
    :return: 预处理后的文档列表
    """
    all_text = []
    all_data = load_files(container_path=root_path, categories=file_path_li, 
    encoding=file_encoding, decode_error="ignore")
    for label, raw_text in zip(all_data.target, all_data.data):
        word_li = preprocess(raw_text)
        label = all_data.target_names[label]
        all_text.append((label, set(word_li)))
    return all_text


def preprocess(raw_text):
    """
    预处理
    :param raw_text:
    :return:
    """
    # 将换行回车符替换为空格
    raw_text = re.sub(u'\r|\n', ' ', raw_text)
    # 去掉数值字母
    raw_text = re.sub(u'[0-9a-zA-z\.]+', u'', raw_text)
    # 分词
    word_li = [w.word for w in NotionalTokenizer.segment(raw_text)]
    # 去除空白符
    word_li = [w.strip() for w in word_li if w.strip()]
    # 移除单字词
    word_li = [w for w in word_li if len(w)>1]
    return word_li


def getVocabulary(all_text):
    """
    获取文档集词汇表
    :param all_text:
    :return:
    """
    global vocabulary
    for label, word_set in all_text:
        vocabulary |= word_set


def multual_infomation(N_10, N_11, N_00, N_01):
    """
    互信息计算
    :param N_10:
    :param N_11:
    :param N_00:
    :param N_01:
    :return: 词项t互信息值
    """
    N = N_11 + N_10 + N_01 + N_00
    I_UC = (N_11 * 1.0 / N) * log2((N_11 * N * 1.0) / ((N_11 + N_10) * (N_11 + N_01))) + \
           (N_01 * 1.0 / N) * log2((N_01 * N * 1.0) / ((N_01 + N_00) * (N_01 + N_11))) + \
           (N_10 * 1.0 / N) * log2((N_10 * N * 1.0) / ((N_10 + N_11) * (N_10 + N_00))) + \
           (N_00 * 1.0 / N) * log2((N_00 * N * 1.0) / ((N_00 + N_10) * (N_00 + N_01)))
    return I_UC


def chi_square(N_10, N_11, N_00, N_01):
    """
    卡方计算
    :param N_10:
    :param N_11:
    :param N_00:
    :param N_01:
    :return: 词项t卡方值
    """
    fenzi = (N_11 + N_10 + N_01 + N_00)*(N_11*N_00-N_10*N_01)*(N_11*N_00-N_10*N_01)
    fenmu = (N_11+N_01)*(N_11+N_10)*(N_10+N_00)*(N_01+N_00)
    if fenmu == 0:
        return 0
    return fenzi*1.0/fenmu


def freq_select(t_doc_cnt, doc_cnt):
    """
    频率特征计算
    :param t_doc_cnt: 类别c中含有词项t的文档数
    :param doc_cnt: 类别c中文档总数
    :return: 词项t频率特征值
    """
    return t_doc_cnt*1.0/doc_cnt


def selectFeatures(documents, category_name, top_k, select_type="chi"):
    """
    特征抽取
    :param documents: 预处理后的文档集
    :param category_name: 类目名称
    :param top_k:  返回的最佳特征数量
    :param select_type: 特征选择的方法，可取值chi,mi,freq，默认为chi
    :return:  最佳特征词序列
    """
    L = []
    # 互信息和卡方特征抽取方法
    if select_type == "chi" or select_type == "mi":
        for t in vocabulary:
            N_11 = 0
            N_10 = 0
            N_01 = 0
            N_00 = 0
            N = 0
            for label, word_set in documents:
                if (t in word_set) and (category_name == label):
                    N_11 += 1
                elif (t in word_set) and (category_name != label):
                    N_10 += 1
                elif (t not in word_set) and (category_name == label):
                    N_01 += 1
                elif (t not in word_set) and (category_name != label):
                    N_00 += 1
                else:
                    print("N error")
                    exit(1)

            if N_00 == 0 or N_01 == 0 or N_10 == 0 or N_11 == 0:
                continue
            # 互信息计算
            if select_type == "mi":
                A_tc = multual_infomation(N_10, N_11, N_00, N_01)
            # 卡方计算
            else:
                A_tc = chi_square(N_10, N_11, N_00, N_01)
            L.append((t, A_tc))
    # 频率特征抽取法
    elif select_type == "freq":
        for t in vocabulary:
            # C类文档集中包含的文档总数
            doc_cnt = 0
            # C类文档集包含词项t的文档数
            t_doc_cnt = 0
            for label, word_set in documents:
                if category_name == label:
                    doc_cnt += 1
                    if t in word_set:
                        t_doc_cnt += 1
            A_tc = freq_select(t_doc_cnt, doc_cnt)
            L.append((t, A_tc))
    else:
        print("error param select_type")
    return sorted(L, key=lambda x:x[1], reverse=True)[:top_k]


# 定义词汇表
vocabulary = set()


def main():
    # 读取文档集（需要根据具体类目名称修改）
    category_name_li = ["Medical", "Sports", "Agriculture",
                        "Education", "Electronics", "Communication"]
    # 获取文本（根目录需要根据具体类目名称修改）
    all_text = getDocuments(r"data/news", category_name_li, "gbk")
    print("all_text len = ", len(all_text))
    # 读取词汇表
    getVocabulary(all_text)
    print("vocabulary len = ", len(vocabulary))
    # 获取特征词表
    print("="*20, '\n', "  卡方特征选择  \n", "="*20)
    feature_select_type = "chi"
    for category_name in category_name_li:
        # 特征抽取，最后一个参数可选值 "chi"卡方,"mi"互信息,"freq"频率
        feature_li = selectFeatures(all_text, category_name, 10, feature_select_type)
        print(category_name)
        for t, i_uc in feature_li:
            print("%s\t%.3f" % (t, i_uc))
        print("="*10)
    
    print("="*20, '\n', "  互信息特征选择  \n", "="*20)
    feature_select_type = "mi"
    for category_name in category_name_li:
        # 特征抽取，最后一个参数可选值 "chi"卡方,"mi"互信息,"freq"频率
        feature_li = selectFeatures(all_text, category_name, 10, feature_select_type)
        print(category_name)
        for t, i_uc in feature_li:
            print("%s\t%.3f" % (t, i_uc))
        print("="*10)

    print("="*20, '\n', "  频率特征选择  \n", "="*20)
    feature_select_type = "freq"
    for category_name in category_name_li:
        # 特征抽取，最后一个参数可选值 "chi"卡方,"mi"互信息,"freq"频率
        feature_li = selectFeatures(all_text, category_name, 10, feature_select_type)
        print(category_name)
        for t, i_uc in feature_li:
            print("%s\t%.3f" % (t, i_uc))
        print("="*10)
    print("program finished")


if __name__ == "__main__":
    main()

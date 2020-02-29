#coding:utf-8
"""

"""

import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from CommonPreprocess import preprocess
import sys


# 文本集的向量化表示
def get_doc_vector(texts_li):
    vectorizer = TfidfVectorizer(max_df=0.4, min_df=2, max_features=20000, analyzer=preprocess)
    text_sets_matrix = vectorizer.fit_transform(texts_li)
    print("文本集向量初始化完毕\n样本数量：%d, 词汇表长度(特征数量)：%d" % text_sets_matrix.shape)
    return vectorizer, text_sets_matrix


# 文本聚类
def kmeans_cluster(text_sets_matrix, n_clusters):
    kmean = KMeans(n_clusters=n_clusters, verbose=0)
    kmean.fit(text_sets_matrix)
    print("聚类结束\n聚类数量：%d，代价：%.2f" % (n_clusters, kmean.inertia_))
    return kmean


# 抽取各簇关键词
def extract_keywords(vectorizer, kmean, topx):
    clusters_keywords_matrix = []
    order_centers = kmean.cluster_centers_.argsort()[:,::-1]
    terms = vectorizer.get_feature_names()
    for i in range(order_centers.shape[0]):
        clusters_keywords_matrix.append([terms[ind] for ind in order_centers[i,:topx]])
    return clusters_keywords_matrix


def main():
    input_filename = sys.argv[1]
    n_clusters = int(sys.argv[2])
    topx = int(sys.argv[3])
    # 读入文本集
    texts_li = []
    with codecs.open(input_filename, 'rb', 'utf-8', 'ignore') as infile:
        texts_li = infile.readlines()
    # 构建文本集向量化矩阵
    vectorizer, text_sets_matrix = get_doc_vector(texts_li)
    # KMenas聚类
    kmean = kmeans_cluster(text_sets_matrix, n_clusters)
    # 抽取簇关键词
    clusters_keywords_matrix = extract_keywords(vectorizer, kmean, topx)
    for keywords_li in clusters_keywords_matrix:
        print(u' '.join(keywords_li))


if __name__ == "__main__":
    main()

#coding:utf-8
"""
Kmeans算法聚类文本示例
"""

import matplotlib.pyplot as plt
import numpy as np

# 加载文本数据
from time import time
from sklearn.datasets import load_files
print("loading documents ...")
t = time()
docs = load_files('../data/cluster_data')
print("summary: {0} documents in {1} categories.".format(len(docs.data), len(docs.target_names)))
print("done in {0} seconds".format(time() - t))

# 文本向量化表示
from sklearn.feature_extraction.text import TfidfVectorizer
max_features = 20000
print("vectorizing documents ...")
t = time()
vectorizer = TfidfVectorizer(max_df=0.4, min_df=2, max_features=max_features, encoding='latin-1')
X = vectorizer.fit_transform((d for d in docs.data))
print("n_samples: %d, n_features: %d" % X.shape)
print("number of non-zero features in sample [{0}]: {1}".format(docs.filenames[0], X[0].getnnz()))
print("done in {0} seconds".format(time() - t))

# 文本聚类
from sklearn.cluster import KMeans, MiniBatchKMeans
print("clustering documents ...")
t = time()
n_clusters = 4
kmean = MiniBatchKMeans(n_clusters=n_clusters, max_iter=100, tol=0.01, verbose=1, n_init=3)
kmean.fit(X)
print("kmean: k={}, cost={}".format(n_clusters, int(kmean.inertia_)))
print("done in {0} seconds".format(time() - t))

# 打印实例数量
print("实例总数 = ", len(kmean.labels_))

# 打印实例1000到1009的簇号
print("1000到1009实例的簇序号：", kmean.labels_[1000:1010])

# 打印实例1000到1009的文件名
print("1000到1009实例的文件名：", docs.filenames[1000:1010])

# 打印每个簇的前10个显著特征
print("每个簇的前10个显著特征：")
order_centroids = kmean.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(n_clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
# coding:utf-8
"""
朴素贝叶斯示例-英文新闻文本分类
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

## 导入数据集以及数据集的简单处理
# 导入全部的训练集和测试集
news = fetch_20newsgroups(subset="all")
# 打印类目名称列表
print("类目名称列表\n",u'\n'.join(news.target_names))
# 打印类目数量
print("类目数量\n", len(news.target_names))
# 打印数据X量
print("训练集文本数量\n", len(news.data))
# 打印类标Y量
print("标记了类别的文本数量\n", len(news.target))
# 打印第0篇文本
print("第1篇文本内容\n", news.data[0])
# 打印类目序号
print("第1篇文本的类别序号\n", news.target[0])
# 打印类目序号所对应的类目名称
print("第1篇文本的类别名称\n", news.target_names[news.target[0]])
# 数据集切分
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, random_state=42)
# 向量化
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)
print("="*50)
# 构建多项式朴素贝叶斯实例
mul_nb = MultinomialNB()
# 训练模型
mul_nb.fit(X_train, y_train)
# 打印测试集上的混淆矩阵
print("混淆矩阵\n", confusion_matrix(y_true=y_test, y_pred=mul_nb.predict(X_test)))
# 打印测试集上的分类报告
print("分类报告\n", classification_report(y_true=y_test, y_pred=mul_nb.predict(X_test)))
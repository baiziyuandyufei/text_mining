# coding:utf-8
"""
朴素贝叶斯垃圾邮件检测
"""
import numpy as np
import csv
import nltk

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import codecs


# 文本预处理
def preprocessing(text):
    # 分词
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    # 去除停用词
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]
    # 移除少于3个字母的单词
    tokens = [word for word in tokens if len(word) >= 3]
    # 大写字母转小写
    tokens = [word.lower() for word in tokens]
    # 词干还原
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def modelbuilding(sms_data, sms_labels):
    """
    构建分类器的流水线示例
    1. 构建训练集和测试集
    2. TFIDF向量化器
    3. 构建朴素贝叶斯模型
    4. 打印准确率和其他评测方法
    5. 打印最相关特征
    :param sms_data:
    :param sms_labels:
    :return:
    """
    # 构建训练集和测试集步骤
    trainset_size = int(round(len(sms_data) * 0.70))
    # 我选择70：30的比例
    print('训练集大小： ' + str(trainset_size) + '\n')
    x_train = np.array([''.join(el) for el in sms_data[0:trainset_size]])
    y_train = np.array([el for el in sms_labels[0:trainset_size]])
    x_test = np.array([''.join(el) for el in sms_data[trainset_size + 1:len(sms_data)]])
    y_test = np.array([el for el in sms_labels[trainset_size + 1:len(sms_labels)]])

    # We are building a TFIDF vectorizer here
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', strip_accents='unicode', norm='l2')
    X_train = vectorizer.fit_transform(x_train)
    X_test = vectorizer.transform(x_test)

    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train, y_train)
    y_nb_predicted = clf.predict(X_test)
    print(y_nb_predicted)

    # 输出测试集上的混淆矩阵
    from sklearn.metrics import confusion_matrix
    print(' \n 混淆矩阵 \n ')
    cm = confusion_matrix(y_test, y_nb_predicted)
    print(cm)

    # 输出测试集上的分类结果报告
    from sklearn.metrics import classification_report
    print('\n 分类报告')
    print(classification_report(y_test, y_nb_predicted))

    # 输出第0个类别上的top10(低频,高频)特征对
    print("输出第0个类别上的top10(低频,高频)特征对")
    # coefs = clf.coef_
    # intercept = clf.intercept_
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    n = 10
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))


if __name__ == '__main__':
    sms_data = []
    sms_labels = []
    with codecs.open('../data/SMSSpamCollection', 'rb', 'utf8', 'ignore') as infile:
        csv_reader = csv.reader(infile, delimiter='\t')
        for line in csv_reader:
            # 添加类别标记
            sms_labels.append(line[0])
            # 添加预处理后的文本
            sms_data.append(preprocessing(line[1]))
    print("原始数据大小：", len(sms_data))
    print("原始标签大小：", len(sms_labels))

    # 我们正则调用model构建函数
    modelbuilding(sms_data, sms_labels)
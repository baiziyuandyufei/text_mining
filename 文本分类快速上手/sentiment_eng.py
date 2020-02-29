# coding:utf-8
"""
朴素贝叶斯影评情感分析
"""
from nltk.corpus import movie_reviews
from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def get_data():
    """
    获取影评数据
    """
    dataset = []
    y_labels = []
    # 遍历类别
    for cat in movie_reviews.categories():
        # 遍历每个类目的评论id
        for fileid in movie_reviews.fileids(cat):
            # 读取评论词语列表
            words = list(movie_reviews.words(fileid))
            dataset.append((words, cat))
            y_labels.append(cat)
    return dataset, y_labels


def get_train_test(input_dataset, ylabels):
    """
    划分数据为训练集和测试集
    """
    train_size = 0.7
    test_size = 1-train_size
    stratified_split = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=77)
    for train_indx, test_indx in stratified_split.split(input_dataset, ylabels):
        train = [input_dataset[i] for i in train_indx]
        train_y = [ylabels[i] for i in train_indx]

        test = [input_dataset[i] for i in test_indx]
        test_y = [ylabels[i] for i in test_indx]

    return train, test, train_y, test_y


def build_word_features(instance):
    """
    构建特征词典
    one-hot, 特征名称为词语本身，特征值为bool类型值
    """
    # 存储特征的词典
    feature_set = {}
    # instance的第1个元素为词语列表
    words = instance[0]
    # 填充特征词典
    for word in words:
        feature_set[word] = 1
    # instance的第2个元素为类别名称
    return feature_set, instance[1]


def build_negate_features(instance):
    """
    如果一个词语前有否定关键词(not或no)修饰，则对词语加前缀Not_, 否定关键词不再被添加到特征词典
    """
    # Retreive words, first item in instance tuple
    words = instance[0]
    final_words = []
    # A boolean variable to track if the 
    # previous word is a negation word
    negate = False
    # List of negation words
    negate_words = ['no', 'not']
    # On looping throught the words, on encountering
    # a negation word, variable negate is set to True
    # negation word is not added to feature dictionary
    # if negate variable is set to true
    # 'Not_' prefix is added to the word
    for word in words:
        if negate:
            word = 'Not_' + word
            negate = False
        if word not in negate_words:
            final_words.append(word)
        else:
            negate = True
    # Feature dictionary
    feature_set = {}
    for word in final_words:
        feature_set[word] = 1
    return feature_set, instance[1]


def remove_stop_words(in_data):
    """
    去除停用词
    Utility function to remove stop words
    from the given list of words
    """
    stopword_list = stopwords.words('english')
    negate_words = ['no', 'not']
    # We dont want to remove the negate words
    # Hence we create a new stop word list excluding
    # the negate words
    new_stopwords = [word for word in stopword_list if word not in negate_words]
    label = in_data[1]
    # Remove stopw words
    words = [word for word in in_data[0] if word not in new_stopwords]
    return words, label


def build_keyphrase_features(instance):
    """
    构建短语特征
    """
    feature_set = {}
    # 应用map迭代器
    instance = remove_stop_words(instance)
    words = instance[0]

    # 使用nltk.collocations的BigramCollocationFinder
    bigram_finder = BigramCollocationFinder.from_words(words)
    # 2grams按词频降序排列，前400个作为关键短语抽取
    bigrams = bigram_finder.nbest(BigramAssocMeasures.raw_freq, 400)
    for bigram in bigrams:
        feature_set[bigram] = 1
    return feature_set, instance[1]


def build_model(features):
    """
    用给定特征集构建朴素贝叶斯模型（NLTK的朴素贝叶斯分类器）
    """
    model = nltk.NaiveBayesClassifier.train(features)
    return model    


def probe_model(model, features, dataset_type='Train'):
    """
    计算测试集准确率, nltk新版里已经没有nltk.classify.accuracy()方法，
    所以这里自己编写precision值
    """
    right_cnt = 0
    sum_cnt = 0

    for feature in features:
        if model.classify(feature[0]) == feature[1]:
            right_cnt += 1
        sum_cnt += 1

    if sum_cnt > 0:
        accuracy = right_cnt * 100.0 / sum_cnt
        print("\n" + dataset_type + " Accuracy = %0.2f" % accuracy + "%")


def show_features(model, no_features=5):
    """
    显示对分类有帮助的特征（NLTK中显示显著特征的方法）
    """
    print("\nFeature Importance")
    print("===================\n")
    print(model.show_most_informative_features(no_features))


def build_model_cycle_1(train_data, dev_data):
    """
    用build_word_features构建特征训练模型
    """
    # Build features for training set
    train_features = map(build_word_features, train_data)
    # Build features for test set
    dev_features = map(build_word_features, dev_data)
    # Build model
    model = build_model(train_features)    
    # Look at the model Python3的map返回的是迭代器，而不是列表，所以在训练使用后想再使用，需要再调用一次
    train_features = map(build_word_features, train_data)
    print("\n词语特征训练集准确率", end='')
    probe_model(model, train_features)
    print("词语特征验证集准确率", end='')
    probe_model(model, dev_features, 'Dev')
    return model


def build_model_cycle_2(train_data, dev_data):
    """
    用build_negate_features构建特征训练模型
    """

    # Build features for training set
    train_features = map(build_negate_features,train_data)
    # Build features for test set
    dev_features = map(build_negate_features,dev_data)
    # Build model
    model = build_model(train_features)    
    # Look at the model
    train_features = map(build_negate_features, train_data)
    print("\n否定词修饰特征训练集准确率", end='')
    probe_model(model, train_features)
    print("否定词修饰特征验证集准确率", end='')
    probe_model(model, dev_features,'Dev')
    
    return model

    
def build_model_cycle_3(train_data, dev_data):
    """
    用build_keyphrase_features构建特征训练模型
    """
    
    # Build features for training set
    train_features = map(build_keyphrase_features, train_data)
    # Build features for test set
    dev_features = map(build_keyphrase_features, dev_data)
    # Build model
    model = build_model(train_features)    
    # Look at the model
    train_features = map(build_keyphrase_features, train_data)
    print("\n2gram特征训练集准确率", end='')
    probe_model(model, train_features)
    print("2gram特征验证集准确率", end='')
    probe_model(model, dev_features, 'Dev')
    test_features = map(build_keyphrase_features, test_data)
    print("2gram特征测试集准确率", end='')
    probe_model(model, test_features, 'Test')
    return model


if __name__ == "__main__":
    
    # Load data
    input_dataset, y_labels = get_data()
    # Train data    
    train_data, all_test_data, train_y, all_test_y = get_train_test(input_dataset, y_labels)
    # Dev data
    dev_data, test_data, dev_y, test_y = get_train_test(all_test_data, all_test_y)

    # Let us look at the data size in our different 
    # datasets
    print("\nOriginal  Data Size   =", len(input_dataset))
    print("\nTraining  Data Size   =", len(train_data))
    print("\nDev       Data Size   =", len(dev_data))
    print("\nTesting   Data Size   =", len(test_data))

    # 用词语特征训练验证模型
    model_cycle_1 = build_model_cycle_1(train_data, dev_data)
    # 打印显著特征
    print("词语显著特征", end='')
    show_features(model_cycle_1)
    # 用否定词修饰的词语特征训练验证模型
    model_cycle_2 = build_model_cycle_2(train_data, dev_data)
    # 打印显著特征
    print("否定词修饰显著特征", end='')
    show_features(model_cycle_2)
    # 用2gram搭配特征训练验证模型
    model_cycle_3 = build_model_cycle_3(train_data, dev_data)
    # 打印显著特征
    print("2gram显著特征", end='')
    show_features(model_cycle_3)
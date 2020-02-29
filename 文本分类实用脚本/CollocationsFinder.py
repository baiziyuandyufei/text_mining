#coding:utf-8
"""
寻找文本固定搭配
"""

import sys
import codecs
from CommonPreprocess import preprocess_single_word
from nltk.collocations import BigramCollocationFinder
import nltk

bigram_measures = nltk.collocations.BigramAssocMeasures()


def main():
    input_filename = sys.argv[1]
    topx = int(sys.argv[2])
    text = u''
    with codecs.open(input_filename, 'rb', 'utf-8', 'ignore') as infile:
        text = infile.read()
    word_li = preprocess_single_word(text)
    finder = BigramCollocationFinder.from_words(word_li)
    print("构建搭配查找器结束")
    collocations_li = finder.nbest(bigram_measures.likelihood_ratio, topx)
    print(u' '.join([u'%s_%s'% w for w in collocations_li]))


if __name__ == "__main__":
    main()

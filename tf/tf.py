#coding:utf8
from __future__ import print_function,unicode_literals
# tf 词频计算，基于 jieba 分词组件
# 由于jieba分词组件并没有计算出词频数据，所以需要自己实现，虽然在jieba内部已经计算过词频，所以计算出来的是已经去重了的。
#这里需要自己计算
import os.path
_context_path = os.path.split(os.path.realpath(__file__))[0]
import re
import chcut.Chcut as Chcut

class TFIDF:
    '''Class TF, To calculate the term frequency'''
    def __init__(self):
        self.chcut = Chcut.Chcut()

    def get_tfidf(self,content):
        tf =  self.chcut.get_term_frequency(content1)
        idf_list =  self.chcut.get_idf(content1, 50)
        tf_idf_list = []
        for idf in idf_list:
            if tf.has_key(idf[0]):
                tf_idf_list.append((idf[0], idf[1] * (tf.get(idf[0]))))

        tf_idf_list.sort(lambda x, y: cmp(x[1], y[1]))
        tf_idf_list.reverse()
        return tf_idf_list
        # for tfidf in tf_idf_list:
        #     print(tfidf)

    def get_tf(self,content):
        return self.chcut.get_term_frequency(content)


    def get_n_of_d(self,content):
        pass


if __name__ == '__main__':
    test1_file_path = "../test/test/test3.txt"
    test2_file_path = "../train/culture/culture-2.txt"
    test1_file = open(test1_file_path,'r')
    test2_file = open(test2_file_path, 'r')
    content1 = [line.strip() for line in test1_file.readlines()]
    content1 = "".join(content1)
    content2 = [line.strip() for line in test2_file.readlines()]
    content2 = "".join(content2)

    # chcut = Chcut.Chcut()
    #
    # tf  = chcut.get_term_frequency(content1)
    # idf_list = chcut.get_idf(content1,50)
    # tf_idf_list = []
    # for idf in idf_list:
    #     if tf.has_key(idf[0]):
    #         tf_idf_list.append((idf[0],idf[1]*(tf.get(idf[0]))))
    #
    # tf_idf_list.sort(lambda x, y: cmp(x[1], y[1]))
    # tf_idf_list.reverse()
    # for tfidf in tf_idf_list:
    #     print (tfidf)

    tfidf = TFIDF()
    # for word,td in tfidf.get_tfidf(content1):
    #     print(word,td)
    class_tf = tfidf.get_tf(content2)
    for tf in class_tf:
        print(tf,class_tf.get(tf))

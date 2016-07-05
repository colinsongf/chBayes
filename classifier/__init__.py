#coding:utf8
from __future__ import print_function,unicode_literals,division
import os
import sys
import chcut.Chcut as Chcut
# 导入遍历组件
import pathWalker
import math
# 导入计时组件
import time
# 导入线程池模块
import multiprocessing
# 导入序列化组件
try:
    import cPickle as pickle
except ImportError:
    import pickle
# 导入tfidf组件
import tfidf
import logging

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

reload(sys)
sys.setdefaultencoding("utf-8")

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

""" The Chinese Classifier Class

    A Naiver Bayes Chinese Web News Classifier
    Which is Build For Graduation Project.

    Author: Chen Quan <tinycq@163.com>
    Date:20160422
    Dependency: Jieba (matplotlib,numpy for Result_handle.py)
    Corpus: Fudan University Chinese News Classify Corpus.
    Ref:
        1. A Comparison of Event Models for Naive Bayes Text Classification (Andrew McCallum, Kamal Nigam)
        2. Transferring Naive Bayes Classifiers for Text Classification (Wenyuan Da. etc.)
        3. Improvement and Application of TFIDF Method Based on Text Classification (ZHANG Yufang. etc.)
        4. A Web Document Classifier Based On Naive Bayes Method:WebCAT(Yu Fang.)
        5. The Simple Implement of Effective Naive Bayes Web News Text Classification Model(Zhihui Wu.etc.)

"""


class Classifier:

    """ The Classifier Class

    """

    def __init__(self):

        """ The Initialized Function for Classifier

            Keyword arguments:
        """

        # 分词器
        self.chcut = Chcut.Chcut()

        # 本程序的根目录，用于定位相对目录
        self._cache_path = _get_abs_path(__file__)

        # 全局测试集和训练集的父目录 "../Parent"
        self.parent_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(self._cache_path),os.path.pardir)),'Parent')

        # 训练集路径
        self.train_set_dir = os.path.join(self.parent_dir, 'Train')

        # 测试集路径 事实上没有用到
        self.test_set_dir = os.path.join(self.parent_dir, 'Test')

        # 分类缓存文件父目录
        self._cache_dir_path = os.path.join(os.path.dirname(self._cache_path),"cache")
        # 判断是否存在cache文件夹
        if not os.path.exists(self._cache_dir_path):
            os.makedirs(self._cache_dir_path)

        # 分类缓存位置
        self.class_tf_cache_dic = None
        # 分类词数缓存目录
        self.class_word_num_cache = 0

        # 全词tf缓存路径
        self._all_word_tf_cache_dir = os.path.join(os.path.dirname(self._cache_path),"all-cache")

        # 判断是否存在全词tf缓存文件夹
        if not os.path.exists(self._all_word_tf_cache_dir):
            os.makedirs(self._all_word_tf_cache_dir)

        # _all_word_tf_cache 全文词频缓存文件 建立全词词频缓存
        self._all_word_tf_cache = os.path.join(self._all_word_tf_cache_dir,"all-tf.cache")

        # _all_word_num_cache 全文词数缓存文件
        self._all_word_num_cache = os.path.join(self._all_word_tf_cache_dir, "all-num.cache")
        default_logger.debug("[CLS] Full Word Cache Path: %s" % self._all_word_tf_cache)

        # 全词词数内存缓存
        self.all_num = 0
        # 全词词频字典内存缓存
        self.all_tf = None

        # 建立全词缓存
        self._get_all_tf()

        # 分类缓存字典
        self.class_word_num_cache_dic = {}

        # 用来定义分类缓存集，从Train中取出
        # self.class_list = ['agriculture', 'economy', 'politics', 'enviornment', 'sports']
        self.class_list = self._get_train_class_list()

        # 建立分类缓存
        self._init_class_cache(self.class_list)
        default_logger.debug("[CLS] Successful Initialized..")

    def _get_train_class_list(self):
        """ get the Class list from The Train Set Dir

            得到Train 训练集中的分类列表
            Key arguments:

        """
        _train_dir = self.train_set_dir
        return os.listdir(_train_dir)

    def _get_all_tf(self):
        """ Build the All Word Cache

            建立全词缓存，便于再次计算
        """
        if os.path.exists(self._all_word_tf_cache) and os.path.isfile(self._all_word_tf_cache):
            default_logger.debug("[CLS] Full Word Cache has been Built.")
        else:
            try:
                # 全词词频
                open(self._all_word_tf_cache, 'wb')
                # 全词数目
                open(self._all_word_num_cache, 'wb')
                # 取得所有文本路径
                _train_paths = pathWalker.pathwalker(self.parent_dir)
                _trainset = _train_paths['train']
                # print(trainset)
                _train_set_files = []
                for cls in _trainset:
                    for _tmp_class_file in os.listdir(cls['classpath']):
                        _train_set_files.append(os.path.join(cls['classpath'], _tmp_class_file))
                # 所有训练集的路径
                print(_train_set_files)
                # 定义全词缓存字典
                _all_tf_cache = {}
                # 开始取词频缓存
                for train_file in _train_set_files:
                    _temp_tf_dic = tfidf.getTF(self._get_file_content(train_file), self.chcut)
                    for _tmp_word in _temp_tf_dic:
                        _tmp_class_tf = 0
                        if _all_tf_cache.get(_tmp_word) is not None:
                            _tmp_class_tf = _all_tf_cache.get(_tmp_word)

                        _tmp_document_tf = _temp_tf_dic.get(_tmp_word)
                        _sum_tf = _tmp_class_tf + _tmp_document_tf

                        if _all_tf_cache.get(_tmp_word) is not None:
                            _all_tf_cache[_tmp_word] = _sum_tf
                        else:
                            _all_tf_cache.update({_tmp_word: _sum_tf})

                # 序列化存储
                print(self._all_word_tf_cache)
                # 序列化存储词频
                _cache_file = open(self._all_word_tf_cache, 'wb')
                pickle.dump(_all_tf_cache, _cache_file)
                _cache_file.close()
                # 序列化存储词的数目
                _cache_word_number_file = open(self._all_word_num_cache, 'wb')
                _word_count_dic = {"all": 0}
                for ctf in _all_tf_cache:
                    _word_count_dic['all'] += _all_tf_cache.get(ctf)
                    # _word_count = class_tf.get(ctf)
                pickle.dump(_word_count_dic, _cache_word_number_file)
                print(_word_count_dic)
                default_logger.debug("[CLS] Cache has been Built Successfully.")


            except:
                # 建立全词缓存出错
                default_logger.debug("[CLS] Cache has been Built Failed!!")
                return

    def _get_class_tf(self, class_dir_path, class_name, reload_cache=False):
        """Build The Classes Word Dict Cache

            用于取得某分类所有词的词频缓存

            Key Arguments
            class_dir_path -- the train set class dir path
            class_name     -- the classname
            reload_cache   -- if reload the class cache or not (default False)

        """
        class_tf = {}
        if not os.path.exists(class_dir_path):
            print("输入的分类路径不合法！")
            return None
        else:
            _cache_file_path = os.path.join(self._cache_dir_path,class_name+"-tf.cache")
            _cache_word_number_file_path = os.path.join(self._cache_dir_path, class_name + "-word-num.cache")
            if os.path.exists(_cache_word_number_file_path) and os.path.exists(_cache_word_number_file_path) and not reload_cache:
                default_logger.debug("[CLS] The Class Cache is Already Existed, skip the Cache Building...")
                return None
            _base_dir_path = os.path.abspath(class_dir_path)
            # 开始遍历文件夹
            files = os.listdir(class_dir_path)
            for file in files:
                full_file_path = os.path.join(_base_dir_path,file)

                _temp_tf_dic = tfidf.getTF(self._get_file_content(full_file_path), self.chcut)
                for _tmp_word in _temp_tf_dic:
                    _tmp_class_tf = 0
                    if class_tf.get(_tmp_word) is not None:
                        _tmp_class_tf = class_tf.get(_tmp_word)

                    _tmp_document_tf = _temp_tf_dic.get(_tmp_word)
                    _sum_tf = _tmp_class_tf + _tmp_document_tf

                    if class_tf.get(_tmp_word) is not None:
                        class_tf[_tmp_word] = _sum_tf
                    else:
                        class_tf.update({_tmp_word: _sum_tf})

            # 序列化存储
            default_logger.debug("[CLS] Class Cache Path: %s" % _cache_file_path)
            #序列化存储词频
            _cache_file = open(_cache_file_path,'wb')
            pickle.dump(class_tf,_cache_file)
            _cache_file.close()
            #序列化存储词的数目
            _cache_word_number_file = open(_cache_word_number_file_path,'wb')
            _word_count_dic  = {class_name:0}
            for ctf in class_tf:
                _word_count_dic[class_name] += class_tf.get(ctf)
                # _word_count = class_tf.get(ctf)
            pickle.dump(_word_count_dic, _cache_word_number_file)
            print(_word_count_dic)
            print("缓存成功...")

    def get_p_word_of_a_class(self, word, class_name):

        """ get the probability of a word in a class
            the Method is very simple: get the term frequency (TF) then divided by sum of words in the class
            取得一个词在某个分类中出现的概率，方法很简单，取得该词在文章中的词频，除以本类中所有词的数目

            Key Arguments:
            word -- the target word
            class_name -- the target class

        """

        _cache_dir_path = self._cache_dir_path
        _cache_class_tf_file_path = os.path.join(_cache_dir_path,class_name+"-tf.cache")
        _cache_class_word_num_file_path  =os.path.join(_cache_dir_path,class_name+"-word-num.cache")
        '''这里每次都要读取一遍文件，肯定时间慢，所以建立内存缓存'''
        #建立内存缓存
        if self.class_tf_cache_dic is None or self._cache_dir_path is None or self.class_word_num_cache.get(word) is None:
            try:
                _cache_class_tf_file = open(_cache_class_tf_file_path, 'r')
                _cache_class_word_num_file = open(_cache_class_word_num_file_path, 'r')
                self.class_tf_cache_dic = pickle.load(_cache_class_tf_file)
                self.class_word_num_cache = pickle.load(_cache_class_word_num_file)
                _cache_class_word_num_file.close()
                _cache_class_tf_file.close()
            except IOError:
                default_logger.error("[CLS] 该分类的缓存文件未建立，请先建立分词缓存！")
        _word_num = self.class_word_num_cache.get(class_name)
        # 古德-图灵估计 在样本不足的情况下对未出现的条目权重打折扣
        _word_tf = 1.0e-3
        if self.class_tf_cache_dic.get(word) is not None:
            _word_tf = self.class_tf_cache_dic.get(word)
        return _word_tf/_word_num

    @staticmethod
    def _get_file_content(file_path):
        """ get the content from a file path

            取得文件路径中文件的内容
            Key Arguments:
            file_path: -- the file path of a target document

        """
        if not os.path.exists(file_path):
            default_logger.error("[CLS] 文件不存在！")
            return None
        else:
            try:
                content = ""
                file = open(file_path, "r")
                for line in file:
                    content += line.strip()
                file.close()
                return content
            except:
                return None

    def _init_class_cache(self,class_list):
        """ Build the Class Cache for the Train Set

            建立训练集中所有的分类缓存
            Key Arguments:
            class_list: -- the classes list (which is Train set contained)
        """
        for cls in class_list:
            self._get_class_tf(os.path.join(self.train_set_dir, cls), cls)
            default_logger.debug("[CLS] Class Path is: %s" % os.path.join(self.train_set_dir, cls))


    def get_the_class_of_a_document(self, filename):
        """ get the class of a document

            取得某文本所述的分类，分类范围应该由分类class_list 给出
            Key Arguments:
            filename: -- The Target File Name Actually Is The Document's Absolute Path
        """

        class_list = self.class_list
        start = time.time()
        _prediction = {}
        for test_cls in class_list:
            _prediction.update(self._get_prediction_of_a_class(test_cls, self._get_file_content(filename)))
        print(_prediction)
        temp = _prediction.popitem()
        max = (temp[0], float((temp[1])))
        while len(_prediction) > 0:
            temp = _prediction.popitem()
            if max[1] < float(temp[1]):
                max = (temp[0], float(temp[1]))
        print(max)
        print('判断分类消耗时间', time.time() - start)
        return max

    def _get_prediction_of_a_class(self, class_name, content):
        """ this Method is for get_the_class_of_a_document

            To Get the Prediction of a document content belongs to a class
            取得某文章的内容属于某类的概率
            Key Arguments:
            class_name: -- the target class
            content:    -- the content of target document
        """

        '''用正比近似算法取得预测值'''
        '''用正比模型取得贝叶斯概率，计算瓶颈在是否要在全集中进行比对'''
        '''因为正比模型取得贝叶斯概率的时候，计算的时候是一个非常稀疏的矩阵，主需要进行降维'''
        prediction = {}
        if self.all_num ==0:
            _all_num_file = open(os.path.join(self._all_word_tf_cache_dir, "all-num.cache"), 'r')
            _all_number = pickle.load(_all_num_file)
            self.all_num = _all_number.get('all')
            _all_num_file.close()

        # 建立分类词频缓存 不要每次都从文件中读取
        if self.class_word_num_cache_dic.get(class_name) is None or self.class_word_num_cache_dic.get(class_name) ==0:
            _class_cache_path = self._cache_dir_path
            _class_cache_file_path = os.path.join(_class_cache_path, class_name + "-word-num.cache")
            _class_cache_file = open(_class_cache_file_path, 'r')
            _num_of_this_class = pickle.load(_class_cache_file)
            self.class_word_num_cache_dic.update({class_name:_num_of_this_class.get(class_name)})
            _class_cache_file.close()

        _p_of_this_class = self.class_word_num_cache_dic.get(class_name) / self.all_num

        prediction.update({class_name: 0.0})

        # 取得P(c)
        P_c = _p_of_this_class

        pre_all = 0

        # 取得全文词频
        _tf_of_a_document = self.chcut.get_term_frequency(content)

        # 取得所有特征值
        _all_set_features = tfidf.getTFIDF(content, self.chcut, K=30)
        # 取得本文的所有特征数目
        n = len(_all_set_features)
        # _all_set_features 是文章中所有特征的总数，这里需要对特征集进行修改为并行计算
        # 降低维度提高效率，采用多线程
        # 需要多全集进行划分，对pre_all进行加锁
        # 定义需要开启并行计算的线程数目
        _thread_num = 2
        # 先对 _all_set_features 进行划分，划分数目等于线程数目
        _thread_features = []
        _all_set_features_len = len(_all_set_features)
        _thread_partition_num = _all_set_features_len//_thread_num
        for i in range(_thread_num):
            temp_dic = {}
            for j in range(_thread_partition_num):
                temp_tuple = _all_set_features.popitem()
                if temp_tuple is None:
                    break
                temp_dic.update({temp_tuple[0]: temp_tuple[1]})
            _thread_features.append(temp_dic)
        while len(_all_set_features) > 0:
            temp_tuple = _all_set_features.popitem()
            _thread_features[-1].update({temp_tuple[0]: temp_tuple[1]})

        # 以下部分是多进程处理部分
        # 定义线程池
        pool = multiprocessing.Pool(processes=_thread_num)
        result = []
        for part_feature in _thread_features:
            result.append(pool.apply_async(_thread_for_get_p_o_a_c_simple, (self, _tf_of_a_document, class_name, part_feature)))
        pool.close()
        pool.join()
        for res in result:
            pre_all +=res.get()
            # print("线程计算结果:::", res.get())
        # print('耗时:', time.time() - start)
        # exit()
        # print("multi_thread",pre_all)
        # pre_all =0

        prediction[class_name] = (math.log(P_c)) / n + pre_all
        return prediction


def _thread_for_get_p_o_a_c_simple(classifier, _tf_of_a_document, class_name, document_features):
    """ The MultiThread Function For Concurrent-Computation

        多线程并行计算方法，由于设计原因必须放在类外面，但是这是提供给类使用的方法

        Key Arguments:
        classifier:         -- A instance of Classifier
        _tf_of_a_document:  -- the terms Frequency of a document (Actually is document's content)
        class_name:         -- the target class name
        document_features:  -- a part of full document features, to shrink the computation times

    """

    _pre_all = 0
    for test_word in document_features:
        if _tf_of_a_document.get(test_word) is None:
            continue
        '''取得某词在某类中的概率'''
        P_wi_c = classifier.get_p_word_of_a_class(test_word, class_name)
        P_w = _tf_of_a_document.get(test_word)/classifier.all_num

        '''
            根据贝叶斯定理，文本分类可以转换为:
            P(C|d) 即已知文章d的情况下，求分类C出现的概率
            但是这个是后验概率，无法直接求得，将d形式化为 SUM<i=1->|V|>(Wi)
            P(C|d) = P(C|(W1,W2,...Wi))<i=1->|V|> 其中V为单篇文章中的特征 |V|为取模
            由贝叶斯定义可知(假设W1,W2,...,Wi之间相互独立)
            P(C|Wi)<i=1->|V|> = MULTI<i=1->|V|>(P(Wi|C)*P(C)/P(Wi))
            两边取自然对数
            log(P(C|Wi)<i=1->|V|>) = SUM<i=1->|V|>log(P(Wi|C))+log(P(C))-log(P(wi))
            <==>

            log(P(C|Wi)<i=1->|V|>) = log(P_c) / |V| + SUM<i=1->|V|>log(P(Wi|C)/P(wi))

            <==>

            在程序中等价于:
            |V| = n = len(_features) 即文本特征的个数
            log(P_c) / |V| <=> log(P_c)/n 在 _get_prediction_of_a_class 方法中得到处理

            _pre_all 即下面的求和部分

            :: SUM 表示求和
            :: MULTI 表示求积
            :: <i=1->N>表示i从1到N

        '''
        _pre_all += math.log(P_wi_c / P_w)
    return _pre_all


if __name__ == '__main__':
    classifitor = Classifier()

    classifitor.get_the_class_of_a_document('../Test/art/C3-Art0003.txt')


#coding:utf8
#基于jieba的中文分词处理程序，主要是加入了自定义的停用词处理和用户字典。
from __future__ import print_function,unicode_literals
import jieba
import jieba.analyse
import os.path
import re
import _compat
_context_path = os.path.split(os.path.realpath(__file__))[0]
import strQB

class Chcut:
    '''中文分词处理组件类，加入了字典和停用词'''
    def __init__(self):
        # 取得当前包路径
        _package_path_ =_context_path
        self._user_dict = _package_path_+os.sep+"dic.data"
        self._user_stword = _package_path_+os.sep+"stword.data"
        #构造停用词列表
        self._stop_word_list = list(line.strip().decode("utf8") for line in open(self._user_stword,'r').readlines())
        # print(self._user_dict,self._user_stword)
        jieba.set_dictionary(self._user_dict)
        jieba.initialize()

    def cut(self,content):
        '''正向快速匹配'''
        content = content.encode('utf8', 'ingore')
        content = strQB.strQ2B(content.decode('utf8'))
        wordlist = list(jieba.cut(content.decode("utf8")))
        #去停用词
        words = list(set(wordlist) - set(self._stop_word_list) - set([' ',' ']))
        return words


    def cuta(self,content):
        '''全词精确匹配'''
        content = content.encode('utf8','ingore')
        content = strQB.strQ2B(content.decode('utf8'))
        #全角转半角
        wordlist =  list(jieba.cut(content.decode("utf8"),cut_all=True))
        # 去停用词
        words = list(set(wordlist) - set(self._stop_word_list))
        return words

    def cutc(self,content):
        jieba.calc()

    def tfidf(self,content):
        _tfidf_list = jieba.analyse.extract_tags(content,topK=100,withWeight=True)
        return _tfidf_list

    def _get_sentences(self,content):
        re_han = re.compile("([\u4E00-\u9FD5]+)", re.U)
        blocks = re_han.split(content)
        for blk in blocks:
                if not blk:
                    continue
                if re_han.match(blk):
                    yield blk
    def get_term_frequency(self,content):
        word_dic = {}
        for word in self._get_sentences(content):
            w = self.cut(word)
            for ws in w:
                if word_dic.get(ws) is not None:
                    temp_dic = {ws: word_dic.get(ws) + 1}
                    word_dic.update(temp_dic)
                else:
                    temp_dic = {ws: 1}
                    word_dic.update(temp_dic)
        return(word_dic)

    def get_idf(self,content,topK=20):
        # print(list(jieba.analyse.extract_tags(content,topK=topK,withWeight=True)))
        return list(jieba.analyse.extract_tags(content,topK=topK,withWeight=True))
    def get_content_features(self,content,topK=20):
        word_dic = {}
        for word in self._get_sentences(content):
            w = self.cut(word)
            for ws in w:
                if word_dic.get(ws) is not None:
                    temp_dic = {ws: word_dic.get(ws) + 1}
                    word_dic.update(temp_dic)
                else:
                    temp_dic = {ws: 1}
                    word_dic.update(temp_dic)
        _features = list(jieba.analyse.extract_tags(content, topK=topK, withWeight=True))
        _result_dic = {}
        for f in _features:
            _result_dic.update({f[0]:word_dic.get(f[0])})
        return _result_dic

if __name__ == '__main__':
    chcut = Chcut()
#coding:utf8
from __future__ import unicode_literals, print_function, division
import classifier


def getTFIDF(content,chcut,K=200):
    """ get the idf value from a document's content and return a dict

        从一篇文章中取得idf值返回一个字典
        Key Arguments
        content -- string 文章内容
        chcut   -- instance of Chcut 分词器
    """

    # 参数判断
    if content is None or chcut is None:
        return None
    # 取得tf
    tf_dic = chcut.get_term_frequency(content)
    ret_dic = {}

    for tf in tf_dic.items():
        ret_dic.update({tf[0]: tf[1]})

    # 计算idf
    _TOPKNUM = 1000

    idf_list = chcut.get_idf(content,_TOPKNUM)

    _max_word_tf = 0
    for word in tf_dic.items():
        if word[1]>_max_word_tf:
            _max_word_tf = word[1]

    # 由TFIDF公式的定义
    # TFIDF(d,w) = (0.5+0.5*(TF(d,w)/MAX(v->len(|V|)TF(d,w<v>)))xlog(N/DF(w)))
    # 转换后的tfidf公式为(0.5+0.5*(TF(d,w)/_max_word_tf)*log(idf(w)))

    # 通常而言 idf 列表会比tf 短 (基于这个假设我们取idf 列表为循环)
    tfidf_list = []

    for word_tuple in idf_list:
        _word = word_tuple[0]
        _word_idf = word_tuple[1]
        _word_tf = tf_dic.get(_word)
        if _word_tf is None:
            continue
        # _tfidf = (0.5+0.5*(_word_tf/_max_word_tf)*math.log(_word_idf))
        _tfidf = _word_tf * _word_idf
        tfidf_list.append(tuple((_word, _word_tf, _word_idf, _tfidf)))
    tfidf_list.sort(lambda x, y: -cmp(x[3], y[3]))


    _ret_dic = {}

    # 取前K个特征用于分类

    for _tfidf in tfidf_list[:K]:

         _ret_dic.update({_tfidf[0]:_tfidf[3]})

    return _ret_dic


def getTF(content,chcut):
    """ get the tf value from a document's content and return a dict

        从一篇文章中取得tf值返回一个字典
        Key Arguments
        content -- string 文章内容
        chcut   -- instance of Chcut 分词器
    """

    # 参数判断
    if content is None or chcut is None:
        return None

    # 取得tf
    tf_dic = chcut.get_term_frequency(content)
    ret_dic = {}

    # 取得 tf
    for tf in tf_dic.items():
        ret_dic.update({tf[0]: tf[1]})

    #计算idf

    _TOPKNUM = 1000
    idf_list = chcut.get_idf(content,_TOPKNUM)

    _max_word_tf = 0
    for word in tf_dic.items():
        if word[1]>_max_word_tf:
            _max_word_tf = word[1]

    tfidf_list = []
    for word_tuple in idf_list:
        _word = word_tuple[0]
        _word_idf = word_tuple[1]
        _word_tf = tf_dic.get(_word)
        if _word_tf is None:
            continue
        # _tfidf = (0.5+0.5*(_word_tf/_max_word_tf)*math.log(_word_idf))
        _tfidf = _word_tf * _word_idf
        tfidf_list.append(tuple((_word, _word_tf, _word_idf, _tfidf)))
    tfidf_list.sort(lambda x, y: -cmp(x[3], y[3]))

    _ret_dic = {}

    for _tfidf in tfidf_list[:200]:
        # print(_tfidf)
        _ret_dic.update({_tfidf[0]: _tfidf[1]})
    return _ret_dic

if __name__ == '__main__':
    classifier = classifier.Classifier()

    content = "".join([line.strip() for line in open('../Test/history/C7-History006.txt').readlines()])
    tf_dic = getTF(content, classifier.chcut)
    tfidf_list = getTFIDF(content, classifier.chcut)
    for tfidf_tuple in tfidf_list.items():
        print(tfidf_tuple)

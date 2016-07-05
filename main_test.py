#coding:utf8
from __future__ import print_function, unicode_literals
import time
import pickle
import os
import pathWalker
# 全局测试
import classifier

if __name__ == '__main__':
    '''
        本文件是主要的测试文件，用于测试贝叶斯分类器的准确率和召回率
        Precision(c) = 所有被正确归为c类的页面/所有被归为c类的页面（错的也算）
        Recall(c)    = 所有被正确归为c类的页面/所有本应被归为c类的页面（不算错的）
    '''
    # 每个分类测试的文章数目
    MAXITEM =200

    # 初始化贝叶斯分类器
    classifitor = classifier.Classifier()
    # 开始计时
    starttime = time.time()
    # 取得测试集路径
    ALLSet = pathWalker.pathwalker("Parent")
    TestSet = ALLSet['test']

    classficiation_list = []
    classficiation_count = {}
    for cls in TestSet:
        clsName = cls.get("classname")
        clsSet = os.listdir(cls.get("classpath"))
        # 取得每个分类的计数
        classficiation_count.update({clsName:len(clsSet)})
        print(clsSet)
        # 测试分类

        test_list1 = ['it', 'education']
        # 存储某一篇测试结果的字典
        for index, clsfilename in enumerate(clsSet):
            if index >=MAXITEM:
                break
            test_file_abs_path = os.path.join(classifitor.test_set_dir+os.sep+cls.get("classname"),clsfilename)
            print(test_file_abs_path)
            # 取得测试文件的绝对路径
            pre_class2 = classifitor.get_the_class_of_a_document(test_file_abs_path)
            temp_dic = {"classfilename": clsfilename, "real_class": cls.get("classname"), "pre_class2":pre_class2[0]}
            classficiation_list.append(temp_dic)
            # break
        # break

    print(classficiation_list)
    result_file_path = "result.txt"
    result_file_path = os.path.abspath(result_file_path)
    result_file = open(result_file_path,"wb")
    pickle.dump(classficiation_list,result_file)

    print("计算耗时", time.time() - starttime)
    print("结果生成，结果路径:", result_file_path)


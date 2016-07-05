# coding:utf-8
from __future__ import print_function
import os
import sys

####################################3
# 请按以下格式安排训练集和测试集
# Parent\
#   Train\
#       class1\
#           1.txt
#           2.txt
#           3.txt
#           ...
#           n.txt
#       class2\
#           1.txt
#           2.txt
#           3.txt
#           ...
#           n.txt
#
#   Test\
#       class1\
#           1.txt
#           2.txt
#           3.txt
#           ...
#       class2\
#           1.txt
#           2.txt
#           ...
#
#   Usage:
#   pathWalker("./Parent/")
############################################


def pathwalker(path):
    if path is None:
        print("warning: path is none!")
        return
    #取得绝对路径
    path = os.path.abspath(path)
    # tupple(dirpath, dirnames, filenames)
    # 深度优先遍历
    testFilesPath  = []
    trainFilesPath = []
    for dirpath,dirnames,filenames in os.walk(path):
        for dirname in dirnames:
            # 测试集
            if dirname == 'Test':
                # print("TEST SET PATH:")
                testAbsPath = os.path.abspath(dirpath+os.sep+dirname)
                # print(testAbsPath)
                for subdirpath, subdirnames, subfilenames in  os.walk(testAbsPath):
                    for subdirname in subdirnames:
                        tempClassPath = {'classname':'','classpath':''}
                        tempClassPath['classname'] = subdirname
                        tempClassPath['classpath'] = os.path.abspath(subdirpath + os.sep + subdirname)
                        testFilesPath.append(tempClassPath)

                # os.listdir(testAbsPath)

            #训练集
            if dirname == "Train":
                # print("TRAIN SET PATH:")
                trainAbsPath = os.path.abspath(dirpath + os.sep + dirname)
                # print(trainAbsPath)
                for subdirpath, subdirnames, subfilenames in os.walk(trainAbsPath):
                    for subdirname in subdirnames:
                        tempClassPath = {'classname': '', 'classpath': ''}
                        tempClassPath['classname'] = subdirname
                        tempClassPath['classpath'] = os.path.abspath(subdirpath + os.sep + subdirname)
                        trainFilesPath.append(tempClassPath)

    # 返回一个字典列表
    paths = {'test':testFilesPath,'train':trainFilesPath}
    return paths


if __name__ == '__main__':
    paths = pathwalker("..\Parent")
    testset = paths['test']
    # print(testset)
    for cls in testset:
        # print(cls['classname'])
        # print(cls['classpath'])
        print(os.listdir(cls['classpath']))
    trainset = paths['train']
    # print(trainset)
    for cls in trainset:
        # print(cls['classname'])
        # print(cls['classpath'])
        print(os.listdir(cls['classpath']))

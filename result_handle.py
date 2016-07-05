#coding:utf8
from  __future__ import print_function,unicode_literals,division
import pickle
import numpy as np
import matplotlib.pyplot as plt
'''
    本程序为结果集处理程序，取得以下结果
        Precision(c) = 所有被正确归为c类的页面(right_c)/所有被归为c类的页面（错的也算）(result_c)
        Recall(c)    = 所有被正确归为c类的页面(right_c)/所有本应被归为c类的页面（不算错的）(should_c)
    ATTENTION:需要修改defined_class_list 为训练集和测试集中的分类，否则将没有结果！
'''

if __name__ == '__main__':
    result_file_path = 'result.txt'
    result_file =open(result_file_path,'r')
    results = pickle.load(result_file)

    defined_class_list = ['agriculture', 'economy', 'politics', 'enviornment', 'sports']

    Precisions = []
    Recalls = []
    F1s = []
    for defined_class in defined_class_list:
        print("当前处理类:",defined_class)
        # 正确被归为C类的页面
        defined_right_c = 0
        # 被归为C类的页面
        defined_result_c = 0
        # 本应该分类为C的页面
        defined_should_c = 0

        for result in results:
            if result.get('pre_class2') == defined_class:
                defined_result_c  +=1
            if result.get('real_class') == defined_class:
                defined_should_c +=1
            if result.get('pre_class2') == result.get('real_class') and result.get('real_class') == defined_class:
                defined_right_c +=1

        # print("defined_right_c",defined_right_c,"defined_result_c",defined_result_c,"defined_should_c",defined_should_c)
        # 查准率
        if defined_result_c == 0:
            Precision = 0
        else:
            Precision = defined_right_c/defined_result_c
        Precisions.append(Precision*100)
        # 回召率
        if defined_should_c ==0:
            Recall=0
        else:
            Recall = defined_right_c/defined_should_c
        Recalls.append(Recall*100)
        if Precision + Recall == 0:
            F1s.append(0)
        else:
            F1s.append(Precision*Recall*2/(Precision+Recall)*100)

        print("Precision(查准率):",Precision)
        print("Recall(回召率):", Recall)

        if Precision + Recall == 0:
            print("F1", 0)
        else:
            print("F1", Precision * Recall * 2 / (Precision + Recall))


    # 绘制直方图
    #########################

    # 转换为touple
    defined_class_tuple = (tuple(defined_class_list))

    N = len(defined_class_tuple)

    Precisions_val = tuple(Precisions)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.2       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, Precisions_val, width, color='b')

    Recalls_val = tuple(Recalls)
    rects2 = ax.bar(ind+width+0.2*width, Recalls_val, width, color='g')

    F1s_val = tuple(F1s)
    rects3 = ax.bar(ind + 2*width+0.4*width, F1s_val, width, color='c')

    # add some
    ax.set_ylabel('Percent(%)')
    ax.set_title('Navie Bayes Text Classification(Test Size 200)')
    ax.set_ylim(0,140)

    # ax.set_xticks(ind+width)
    ax.set_xticks(ind+1.7*width)
    ax.set_xticklabels(defined_class_tuple)

    ax.legend((rects1[0], rects2[0],rects3[0]), ('Precision', 'Recall','F1'))

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.00*height, '%d'%int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)


    plt.show()
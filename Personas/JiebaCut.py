import pandas as pd
import jieba.analyse
import time
import jieba
import jieba.posseg
import os, sys

def input(train_name):
    train_data = []
    with open(train_name, 'rb') as f:
        line = f.readline()
        count = 0
        while line:
            try:
                train_data.append(line)
                count += 1
            except:
                print('error', line, count)
            line = f.readline()
    return train_data



testfile_path = r'E:\DM_Operation\Personas\data\test_queryList.csv'
testwrite_path = r'E:\DM_Operation\Personas\data\test_queryList_writefile-1w.csv'

trainfile_path = r'E:\DM_Operation\Personas\data\train_queryList.csv'
trainwrite_path = r'E:\DM_Operation\Personas\data\train_queryList_writefile-1w.csv'

start_time = time.perf_counter()

def cut(path1, path2):
    QueryList = input(path1)
    csvfile = open(path2, 'w')
    POS = {}
    for i in range(len(QueryList)):
        if ((i % 2000 == 0) and (i >= 1000)):
            print(i,'finished')
        s = []
        str = ''
        #词性分词
        words = jieba.posseg.cut(QueryList[i])  #jieba的posseg.cut把第一列数据切分
        allowPOS = ['n', 'v', 'j']      #设置允许通过的词性列表
        for word, flag in words:    #word存词，flag存词性
            c = POS.get(flag, 0) + 1
            #从POS中获取当前词性的出现次数（flag搜索词性，0取次数） 更新旧的次数
            POS[flag] = POS.get(flag, 0) + 1

            if ((flag[0] in allowPOS) and (len(word) >= 2)):    #满足筛选条件，以空格合并
                str += word + " "
        cur_str = str.encode('utf8')    #utf8编码
        cur_str = cur_str.decode('utf8')
        s.append(cur_str)       #完成一行
        v = " ".join(s) + '\n'
        csvfile.write("".join(s) + '\n')
    csvfile.close()

cut(testfile_path, testwrite_path)
cut(trainfile_path, trainwrite_path)


end_time = time.perf_counter()
print ("total time: %f s" % (end_time - start_time))


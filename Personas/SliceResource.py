'''
用户画像-案例
基于用户搜索关键词数据为用户打上标签（年龄，性别，学历）

整体流程
（一）数据预处理
*编码方式转换
*对数据搜索内容进行分词
*词性过滤
*数据检查

（二）特征选择
*建立word2vec词向量模型
*对所有搜索数据求平均向量

（三）建模预测
*不同机器学习模型对比
*堆叠模型
'''

import csv
import os
import pandas as pd

'''训练集切分和编码转换'''
train_path = r'E:\DM_Operation\Personas\data\user_tag_query.10W.TRAIN'
csvfile = open(train_path + '-1w.csv', 'w')
writer = csv.writer(csvfile)
writer.writerow(['ID', 'age', 'Gender', 'Education', 'QueryList'])
print(os.getcwd())
with open(r'E:\pyworkspace\用户画像\data\user_tag_query.10W.TRAIN', 'r', encoding='gb18030', errors='ignore') as f:
    lines = f.readlines()
    for line in lines[0:10000]:
        try:
            line = line.strip()    #空格去掉
            data = line.split('\t')
            writedata = [data[0], data[1], data[2], data[3]]
            querystr = ''
            # data[-1] = data[-1][:-1]
            for d in data[4:]:
                try:
                    cur_str = d.encode('utf8')
                    cur_str = cur_str.decode('utf8')
                    querystr += cur_str + '\t'
                except:
                    continue
            # print(querystr==querystr[:-1])
            querystr = querystr[:-1]
            writedata.append(querystr)
            writer.writerow(writedata)
        except:
            # print (data[0][0:20])
            continue


'''测试集切分和编码转换'''
test_path = r'E:\DM_Operation\Personas\data\user_tag_query.10W.TEST'
csvfile = open(test_path + '-1w.csv', 'w')
writer = csv.writer(csvfile)
writer.writerow(['ID', 'QueryList'])
with open(r'E:\pyworkspace\用户画像\data\user_tag_query.10W.TEST', 'r',encoding='gb18030',errors='ignore') as f:
    lines = f.readlines()
    for line in lines[0:10000]:
        try:
            data = line.split("\t")
            writedata = [data[0]]
            querystr = ''
            data[-1]=data[-1][:-1]
            for d in data[1:]:
                try:
                    cur_str = d.encode('utf8')
                    cur_str = cur_str.decode('utf8')
                    querystr += cur_str + '\t'
                except:
                    #print (data[0][0:10])
                    continue
            querystr = querystr[:-1]
            writedata.append(querystr)
            writer.writerow(writedata)
        except:
            #print (data[0][0:20])
            continue

print('finish')
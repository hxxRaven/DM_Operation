'''
增加专属词汇
分词
去停用词
'''

import pandas as pd
import numpy as np
import csv
import re
import os, time
import jieba
import pickle
from numpy import log
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
data_path = 'data'
jobinfo_path = 'job293.pkl'
df_path = 'df293.pkl'

df = pickle.load(open(data_path+os.sep+df_path, 'rb'))
jobinfo = pickle.load(open(data_path+os.sep+jobinfo_path, 'rb'))
# jobinfo = pd.read_csv(data_path + os.sep + jobinfo_path, index_col=0)
# df = pd.read_csv(data_path+os.sep+df_path, index_col=0)

'''
针对本国家电网相关数据，需要添加一些特定词汇，防止jieba分词器不识别
'''
mywords = ['户号', '分时', '抄表',
           '抄表示数', '工单', '单号',
           '工单号', '空气开关', '脉冲灯',
           '计量表', '来电', '报修']
for word in mywords:
    jieba.add_word(word)

#停用词
stop = set()
with open(data_path+os.sep+'stopwords.txt', encoding='utf-8') as f:
    for word in f:
        word = word.strip()
        stop.add(word)

#分词脚本
def jcut(line):
    res = []
    words = jieba.cut(line)
    for word in words:
        if word not in stop:
            res.append(word)
    return ' '.join(res)

print('开始分词')
st = time.perf_counter()
jobinfo['content'] = jobinfo['ACCEPT_CONTENT'].apply(lambda x : jcut(x))
et = time.perf_counter()
print('结束。用时：', et-st)


'''
手机号，户号等后面连接的号码是不用的，统一一下，具体号码没啥用
用正则匹配之后，把匹配内容统一替换为手机号、户号、工单号等文字
'''

def sub_no(x):
    shouji_pattern = re.compile('\s1\d{10}\s|\s1\d{10}\Z')
    if shouji_pattern.findall(x):
        x = re.sub(shouji_pattern, ' 手机number ', x)

    huhao_pattern = re.compile('\s\d{10}\s|\s\d{10}\Z')
    if huhao_pattern.findall(x):
        x = re.sub(huhao_pattern, ' 户号number ', x)

    tuiding_pattern = re.compile('\s\d{11}\s|\s\d{11}\Z')
    if tuiding_pattern.findall(x):
        x = re.sub(tuiding_pattern, ' 退订number ', x)

    gongdan_pattern = re.compile('\s201\d{13}\s|\s201\d{13}\Z')
    if gongdan_pattern.findall(x):
        x = re.sub(gongdan_pattern, ' 工单number ', x)

    tingdian_pattern = re.compile('\s\d{12}\s|\s\d{12}\Z')
    if tingdian_pattern.findall(x):
        x = re.sub(tingdian_pattern, ' 停电number ', x)

    return x.strip()

print(jobinfo[['content']])

jobinfo['contents'] = jobinfo['content'].apply(lambda x : sub_no(x))
jobinfo['len_of_contents'] = jobinfo['contents'].apply(lambda x : len(x.split()))   #长度
jobinfo['counts_of_words'] = jobinfo['contents'].apply(lambda x : len(set(x.split())))  #几个词

text = df[['CUST_NO']].copy()
text = text.merge(jobinfo[['CUST_NO', 'len_of_contents', 'counts_of_words', 'contents']], on='CUST_NO', how='left')


print(df.columns)
print(df.shape)

pickle.dump(text, open(data_path + os.sep + 'text_features_1.pkl', 'wb'))
pickle.dump(jobinfo, open(data_path + os.sep + 'job294.pkl', 'wb'))
import pandas as pd
import numpy as np
import csv
import re
import os
import jieba
import pickle
from numpy import log
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

data_path = 'rawdata'   #元数据目录
file_jobinfo_train = '01_arc_s_95598_wkst_train.tsv' #完整工单训练集
file_jobinfo_test = '01_arc_s_95598_wkst_test.tsv' #完整工单测试集
file_comm = '02_s_comm_rec.tsv' #通话信息记录
file_flow_train = '09_arc_a_rcvbl_flow.tsv' #应收电费工单训练集
file_flow_test = '09_arc_a_rcvbl_flow_test.tsv' #应收电费工单测试集
file_tlabel = 'train_label.csv' #训练集正例
file_test = 'test_to_predict.csv'   #测试集

train_info = pd.read_csv(data_path + os.sep + 'processed_' + file_jobinfo_train, sep='\t', encoding="utf-8", quoting=csv.QUOTE_NONE)
train_info = train_info.loc[~train_info.CUST_NO.isnull()]
train_info['CUST_NO'] = train_info.CUST_NO.astype(np.int64)
#train为train大表中CUST_NO统计次数表
train = train_info.CUST_NO.value_counts().to_frame().reset_index()  #CUST_NO为用户id，做出用户工单提交次数排名表
# print(train)
train.columns = ['CUST_NO', 'counts_of_jobinfo']

#train_label存放的是正例的用户id，检查train表中的用户是否在正例label（此文件只和train大表有关）中出现(1正例，0负例)
temp = pd.read_csv(data_path + os.sep + file_tlabel, header=None)
temp.columns = ['CUST_NO']
train['label'] = 0
train.loc[train.CUST_NO.isin(temp.CUST_NO), 'label'] = 1
train = train[['CUST_NO', 'label', 'counts_of_jobinfo']]
# print(train)

#测试集label为-1
test_info = pd.read_csv(data_path + os.sep + 'processed_' +file_jobinfo_test, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
test_info = test_info.loc[~test_info['CUST_NO'].isnull()]
#test为test大表中CUST_NO统计次数表，label-1，标记为测试样本
test = test_info['CUST_NO'].value_counts().to_frame().reset_index()
test.columns = ['CUST_NO', 'counts_of_jobinfo']
test['label'] = -1
test = test[['CUST_NO', 'label', 'counts_of_jobinfo']]
# print(test)

'''分两条路走，工单数为1，大于1'''
df = train.append(test).copy()
print(df)
del(temp, train, test)

df = df.loc[df.counts_of_jobinfo == 1].copy()
df.reset_index(drop=True, inplace=True) #drop是否丢掉原索引，inplace是在原frame上修改，还是新起一个frame
train = df.loc[df['label'] != -1]
test = df.loc[df['label'] == -1]

print('原始数据中的低敏感度用户分布情况如下：')
print('训练集：',train.shape[0])
print('正样本:',train.loc[train.label == 1].shape[0])
print('负样本:',train.loc[train.label == 0].shape[0])
print('-----------------------')
print('测试集：',test.shape[0])
df.drop(['counts_of_jobinfo'], axis=1, inplace=True)    #用完即去

'''把总表(总训练+总测试)的相关数据筛选出来'''
jobinfo = train_info.append(test_info).copy()
jobinfo = jobinfo.loc[jobinfo['CUST_NO'].isin(df['CUST_NO'])].copy()
jobinfo.reset_index(drop=True, inplace=True)
jobinfo =jobinfo.merge(df[['CUST_NO', 'label']], on='CUST_NO', how='left')
print(jobinfo)


'''现在处理通话数据'''
comm = pd.read_csv(data_path + os.sep + file_comm, sep='\t')
comm.drop_duplicates(inplace=True)  #去重
print(comm.shape)
#过滤掉没出现在jobinfo的数据
comm = comm.loc[comm['APP_NO'].isin(jobinfo['ID'])]
comm = comm.rename(columns={'APP_NO':'ID'}) #总表中的ID与comm表中的APP_NO是一样的
comm = comm.merge(jobinfo[['ID','CUST_NO']], on='ID', how='left')
print("通话可用数据：", comm.shape)


#通话时间作为常见错误先进性排查(以开始时间大于等于结束时间)
comm['REQ_BEGIN_DATE'] = comm['REQ_BEGIN_DATE'].apply(lambda x : pd.to_datetime(x)) #转化时间格式
comm['REQ_FINISH_DATE'] = comm['REQ_FINISH_DATE'].apply(lambda x : pd.to_datetime(x))
comm = comm.loc[~(comm['REQ_BEGIN_DATE']>=comm['REQ_FINISH_DATE'])]
print('时间清理后的可用数据：',comm.shape)
df = df.loc[df['CUST_NO'].isin(comm['CUST_NO'])].copy() #df作为统计低敏的表格，也要根据用户可用通话记录来更新下

#建立特征：通话时间
comm['holding_time'] = comm['REQ_FINISH_DATE'] - comm['REQ_BEGIN_DATE']
comm['holding_time_seconds'] = comm['holding_time'].apply(lambda x : x.seconds)
df = df.merge(comm[['CUST_NO', 'holding_time_seconds']], how='left', on='CUST_NO')
#归一化秒到[0,1]
df['holding_time_seconds'] = MinMaxScaler().fit_transform(df['holding_time_seconds'].values.reshape(-1, 1))

del(comm)
print(df.head())

# df.to_csv('data' + os.sep + 'df_step1.csv', index=True, header=True)
# jobinfo.to_csv('data' + os.sep + 'joininfo_step1.csv', index=True, header=True)

pickle.dump(df, open('../myfeatures/df291.pkl', 'wb'))
pickle.dump(jobinfo, open('../myfeatures/job291.pkl.pkl', 'wb'))
print('Step1 Fininshed')
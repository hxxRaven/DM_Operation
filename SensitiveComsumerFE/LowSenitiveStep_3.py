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

data_path = 'data'
df_path = 'df292.pkl'
jobinfo_path = 'job292.pkl'

df = pickle.load(open(data_path+os.sep+df_path, 'rb'))
jobinfo = pickle.load(open(data_path+os.sep+jobinfo_path, 'rb'))
# df = pd.read_csv(data_path + os.sep + df_path, index_col=0)
# jobinfo = pd.read_csv(data_path + os.sep + jobinfo_path, index_col=0)
# print(df.head())
# print(jobinfo.head())

'''
收费信息表处理
'''
file_flow_train = '09_arc_a_rcvbl_flow.tsv' #应收电费工单训练集
file_flow_test = '09_arc_a_rcvbl_flow_test.tsv' #应收电费工单测试集

train_flow = pd.read_csv('rawdata' + os.sep + file_flow_train, sep='\t')
test_flow = pd.read_csv('rawdata' + os.sep + file_flow_test, sep='\t')
# print(train_flow)
# print(test_flow)
flow = train_flow.append(test_flow).copy()
flow = flow.rename(columns={'CONS_NO':'CUST_NO'})  #实际为jobinfo的CUST_NO同一属性
flow = flow.drop_duplicates()
# print(flow.head())

#纠正某些不可能为负数的值
flow['T_PQ'] = flow['T_PQ'].apply(lambda x : -x if x <0 else x)
flow['RCVBL_AMT'] = flow.RCVBL_AMT.apply(lambda x : -x if x < 0 else x)
flow['RCVED_AMT'] = flow.RCVED_AMT.apply(lambda x : -x if x < 0 else x)
flow['OWE_AMT'] = flow.OWE_AMT.apply(lambda x : -x if x < 0 else x)

#将收费相关特征存入df
#把收费表中的收取费用按CUST_NO来groupby(),在计算总和，映射到所有此CUST_NO的样本上（+1是防止log变化取负数）
#应收金额
df['sum_yingshoujine'] = log(df['CUST_NO'].map(flow.groupby('CUST_NO').RCVBL_AMT.sum()) + 1)
df['mean_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.mean()) + 1)
df['max_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.max()) + 1)
df['min_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.min()) + 1)
df['std_yingshoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_AMT.std()) + 1)
# 实收金额
df['sum_shishoujine'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_AMT.sum()) + 1)
# 少交了多少（欠费）
df['qianfei'] = df['sum_yingshoujine'] - df['sum_shishoujine']
# 总电量
df['sum_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.sum()) + 1)
df['mean_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.mean()) + 1)
df['max_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.max()) + 1)
df['min_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.min()) + 1)
df['std_T_PQ'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').T_PQ.std()) + 1)
# 电费金额
df['sum_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.sum()) + 1)
df['mean_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.mean()) + 1)
df['max_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.max()) + 1)
df['min_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.min()) + 1)
df['std_OWE_AMT'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').OWE_AMT.std()) + 1)
# 电费金额和应收金额差多少
df['dianfei_jian_yingshoujine'] = df['sum_OWE_AMT'] - df['sum_yingshoujine']
# 应收违约金
df['sum_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.sum()) + 1)
df['mean_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.mean()) + 1)
df['max_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.max()) + 1)
df['min_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.min()) + 1)
df['std_RCVBL_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVBL_PENALTY.std()) + 1)
# 实收违约金
df['sum_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.sum()) + 1)
df['mean_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.mean()) + 1)
df['max_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.max()) + 1)
df['min_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.min()) + 1)
df['std_RCVED_PENALTY'] = log(df.CUST_NO.map(flow.groupby('CUST_NO').RCVED_PENALTY.std()) + 1)
df['chaduoshao_weiyuejin'] = df['sum_RCVBL_PENALTY'] - df['sum_RCVED_PENALTY']

#每个月工单次数
df['has_biao9'] = 0
#df表里有，金额单里也有的标记为1，证明其有消费记录。否则为0
df.loc[df['CUST_NO'].isin(flow.CUST_NO), 'has_biao9'] = 1
#存入金额单中总订单数
df['counts_of_09flow'] = df['CUST_NO'].map(flow.groupby('CUST_NO').size())

# 每个用户有几个月的记录
#groupby了CUST_NO，RCVBL_YM工单号有几次不同，则代表有几个月有次记录
df['nunique_RCVBL_YM'] = df['CUST_NO'].map(flow.groupby('CUST_NO').RCVBL_YM.nunique())

#平均每个月工单数
df['mean_RCVBL_YM'] = df['counts_of_09flow'] / df['nunique_RCVBL_YM']
del(train_flow, test_flow, flow)

print(df.shape)
print(df.columns)

# if not os.path.isdir('../myfeatures'):
#     os.makedirs('../myfeatures')
# pickle.dump(df, open('../myfeatures/statistical_features_1.pkl', 'wb'))

# df.to_csv('data'+os.sep+'statistical_features_1.csv', index=True, header=True)
pickle.dump(df, open(data_path+os.sep+'df293.pkl', 'wb'))
pickle.dump(jobinfo, open(data_path+os.sep+'job293.pkl', 'wb'))
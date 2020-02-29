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
df_path = 'df291.pkl'
jobinfo_path = 'job291.pkl'

df = pickle.load(open(data_path+os.sep+df_path, 'rb'))
jobinfo = pickle.load(open(data_path+os.sep+jobinfo_path, 'rb'))

# df = pd.read_csv(data_path + os.sep + df_path, index_col=0)
# jobinfo = pd.read_csv(data_path + os.sep + jobinfo_path, index_col=0)
# print(df.head())
# print(df.shape)
# print(jobinfo.head())
# print(jobinfo.shape)

'''
df更新后，对应jobinfo也需要更新
'''
jobinfo = jobinfo.loc[jobinfo['CUST_NO'].isin(df['CUST_NO'])].copy()
jobinfo.reset_index(drop=True, inplace=True)

'''
rank函数会对Series进行排名操作，也就是当前数据是排名第几的，指定max方法就是有相同的取靠后的名次。
'''
df['rank_CUST_NO'] = df['CUST_NO'].rank(method='max')
#再把rank值归一化，看下当前样本在总样本中能排在那个位置
df['rank_CUST_NO'] = MinMaxScaler().fit_transform(df['rank_CUST_NO'].values.reshape(-1, 1))
print(df.head())

'''
对离散数据BUSI_TYPE_CODE进行one-hot编码
'''
df = df.merge(jobinfo[['CUST_NO', 'BUSI_TYPE_CODE']], on='CUST_NO', how='left')
temp = pd.get_dummies(df['BUSI_TYPE_CODE'], prefix='onehot_BUST_TYPE_CODE', dummy_na=True)  #dummy_na是否添加一列指示NANS
df = pd.concat([df, temp], axis=1)
df.drop(['BUSI_TYPE_CODE'], axis=1, inplace=True)
print(df.head())
del(temp)

#URBAN_RURAL_FLAG
df = df.merge(jobinfo[['CUST_NO', 'URBAN_RURAL_FLAG']], on='CUST_NO', how='left')
temp = pd.get_dummies(df.URBAN_RURAL_FLAG, prefix='onehot_URBAN_RURAL_FLAG', dummy_na=True)
df = pd.concat([df, temp], axis=1)
df.drop(['URBAN_RURAL_FLAG'], axis=1, inplace=True)
del(temp)

'''
对数据供电单位编码,按长度编码(同类型单位编码长度一样，所以采用长度编码)
'''
df = df.merge(jobinfo[['CUST_NO', 'ORG_NO']], on='CUST_NO', how='left')
df['len_of_ORG_NO'] = df['ORG_NO'].apply(lambda x : len(str(x)))
print(df['len_of_ORG_NO'].isnull().any())
df.fillna(-1, inplace=True)

#对len_of_ORG_NO种类特征，进行onehot编码
temp = pd.get_dummies(df['len_of_ORG_NO'], prefix='onehot_len_of_ORG_NO')
df = pd.concat([df, temp], axis=1)
df.drop(['len_of_ORG_NO'], axis=1, inplace=True)

'''
ration为每个供电单位正例占全部的比重
'''
#ration的结构{'ORG_NO特征种类':这种ORG_NO的正例占全部正负和的比例}
ration = {}
train = df.loc[df['label'] != -1]
for i in train['ORG_NO'].unique():
    ration[i] = len(train.loc[(train['ORG_NO'] == i) & (train['label'] == 1)]) / len(train.loc[train['ORG_NO']==i])
df['ration_ORG_NO'] = df['ORG_NO'].map(ration)
# print(df['ration_ORG_NO'].head())
df.drop(['ORG_NO'], axis=1, inplace=True)


'''时间数据，从data图中，咳哟看到明显区别（随天数变化，工单接收量的变化（HANDLE_TIME））'''
df = df.merge(jobinfo[['CUST_NO', 'HANDLE_TIME']], on='CUST_NO', how='left')
df['date'] = df['HANDLE_TIME'].apply(lambda x : pd.to_datetime(x.split()[0]))   #年月日
df['time'] = df['HANDLE_TIME'].apply(lambda x : x.split()[1])   #时分秒
df['month'] = df['date'].apply(lambda x : x.month)  #月份
df['day'] = df['date'].apply(lambda x : x.day)  #天数
df['hour'] = df['time'].apply(lambda x : int(x.split(':')[0]))  #小时
df.drop(['HANDLE_TIME', 'date', 'time'], axis=1, inplace=True)
features = ['CUST_NO','month','day', 'hour']
print(df[features].head())  #[[]]才能输出

#上中下旬统计
df['inEarly'] = 0
df['inMid'] = 0
df['inLater'] = 0
df.loc[df['day'].isin(range(1,11)) , 'inEarly'] = 1
df.loc[df['day'].isin(range(11,21)) , 'inMid'] = 1
df.loc[df['day'].isin(range(21,32)) , 'inLater'] = 1


'''处理ELEC_TYPE，首位表示用电方式，如工业用电'''
df = df.merge(jobinfo[['CUST_NO', 'ELEC_TYPE']], on='CUST_NO', how='left')
df.fillna(0,inplace=True)
df['head_of_ELEC_TYPE'] = df['ELEC_TYPE'].apply(lambda x : str(x)[0])   #数字转字符串，取第一位
df['is_ELEC_TYPE_NaN'] = 0
df.loc[df['ELEC_TYPE'] == 0, 'is_ELEC_TYPE_NaN'] = 1 #ELECT_TYPE为0，标记

#LabelEncoder(),把列的种类按出现顺序统计，给定连续编号，再用编号表示原有值
df['label_encoder_ELEC_TYPE'] = LabelEncoder().fit_transform(df['ELEC_TYPE'])

#在意引用ration比例,敏感用户和非敏感用户，在用电方式的占比（用电方式正例占整体用电方式的比例）
train = df[df['label'] != -1]
ration = {}
for i in train['ELEC_TYPE'].unique():
    ration[i] = len(train.loc[(train['ELEC_TYPE'] == i) & (train['label'] == 1)]) / len(train.loc[train['ELEC_TYPE'] == i])
df['ration_ELEC_TYPE'] = df['ELEC_TYPE'].map(ration)
df.fillna(0, inplace=True)
print(df[['ration_ELEC_TYPE','head_of_ELEC_TYPE']])

#用电方式onehot
temp = pd.get_dummies(df['head_of_ELEC_TYPE'], prefix='onehot_head_of_ELEC_TYPE')
df = pd.concat([df, temp] ,axis=1)
df.drop(['ELEC_TYPE', 'head_of_ELEC_TYPE'], axis=1, inplace=True)


'''
城市编码，先ration，再onehot
'''
df = df.merge(jobinfo[['CUST_NO', 'CITY_ORG_NO']], on='CUST_NO', how='left')
train = df[df['label'] != -1]
ration = {}
for i in df['CITY_ORG_NO'].unique():
    ration[i] = len(train.loc[(train['CITY_ORG_NO'] == i) & (train['label'] == 1)]) / len(train.loc[train['CITY_ORG_NO'] == i])
df['ration_CITY_ORG_NO'] = df['CITY_ORG_NO'].map(ration)
temp = pd.get_dummies(df['CITY_ORG_NO'], prefix='onehot_CITY_ORG_NO')
df = pd.concat([df, temp], axis=1)
df.drop(['CITY_ORG_NO'], axis=1, inplace=True)

print(df.shape)
print(df.columns)
# df.to_csv('data'+os.sep+'df_step2.csv', index=True, header=True)
# jobinfo.to_csv('data' + os.sep + 'jobinfo_step2.csv', index=True, header=True)
pickle.dump(df, open(data_path+os.sep+'df292.pkl', 'wb'))
pickle.dump(jobinfo, open(data_path+os.sep+'job292.pkl', 'wb'))
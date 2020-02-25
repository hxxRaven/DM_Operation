import pandas as pd
import os

test_path = 'InsuranceForcast_xgboost.data' + os.sep + 'user_tag_query.10W.TEST-1w.csv'
train_path = 'InsuranceForcast_xgboost.data' + os.sep + 'user_tag_query.10W.TRAIN-1w.csv'

data = pd.read_csv(train_path, encoding='gbk')
print(data.info())

#生成训练集性别、年龄、学历label
data['age'].to_csv(r'E:\DM_Operation\Personas\data\train_age.csv' ,index=False)
data['Gender'].to_csv(r'E:\DM_Operation\Personas\data\train_gender.csv' ,index=False)
data['Education'].to_csv(r'E:\DM_Operation\Personas\data\train_education.csv' ,index=False)

#特征数据取出
data['QueryList'].to_csv(r'E:\DM_Operation\Personas\data\train_queryList.csv' ,index=False)

#生成测试集的特征
data = pd.read_csv(test_path, encoding='gbk')
data['QueryList'].to_csv(r'E:\DM_Operation\Personas\data\test_queryList.csv' ,index=False)


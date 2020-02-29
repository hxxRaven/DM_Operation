'''
我们做一个简短的数据探索，看看我们有什么样的数据集，以及我们是否能找到其中的任何模式。
主要使用Xgboost
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import cross_val_score

from scipy import stats
import seaborn as sns
from copy import deepcopy

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


train_data_path = 'data' + os.sep + 'train.csv'
test_data_path = 'data' + os.sep + 'test.csv'


train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)

print(train.shape)
print ('First 20 columns:', list(train.columns[:20]))
print ('Last 20 columns:', list(train.columns[-20:]))

print(train.describe())
#数据已经被预处理了，max，min和均值都接近标准
print(train.info())

print(pd.isnull(train).values.sum())    #缺失值也没有，可以下一步了

#查看object特征下，每个特征的值有几种类别
cat_features = list(train.select_dtypes(include=['object']).columns)

#查看int64特征下，每个特征的值有几种类别
int_features = list(train.select_dtypes(include=['int64']).columns)

#查看float64特征下，每个特征的值有几种类别
cont_features = [cont for cont in list(train.select_dtypes(
    include=['float64', 'int64']).columns) if cont not in ['loss', 'id']]

cat_uniques = []
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))

uniq_values_in_categories = pd.DataFrame.from_dict({'cat_names':cat_features, 'unique_values':cat_uniques})
# print(uniq_values_in_categories)

#看下各种值的分布
print(uniq_values_in_categories['unique_values'].value_counts)
#脚本
fig, ax = plt.subplots(1,2, figsize=(16, 5))
ax[0].hist(uniq_values_in_categories.unique_values, bins=50)
ax[0].set_title('Amount of categorical features with X distinct values')
ax[0].set_xlabel('Distinct values in a feature')
ax[0].set_ylabel('Features')
ax[0].annotate('A feature with 326 vals', xy=(322, 2), xytext=(200, 38), arrowprops=dict(facecolor='black'))
ax[1].set_xlim(2,30)
ax[1].set_title('Zooming in the [0,30] part of left histogram')
ax[1].set_xlabel('Distinct values in a feature')
ax[1].set_ylabel('Features')
ax[1].grid(True)
ax[1].hist(uniq_values_in_categories[uniq_values_in_categories.unique_values <= 30].unique_values, bins=30)
ax[1].annotate('Binary features', xy=(3, 71), xytext=(7, 71), arrowprops=dict(facecolor='black'))   #xy箭头，xytext文本，最后为颜色
plt.show()


#展示赔偿值的情况，以id为横坐标
fig1, ax1 =plt.subplots(figsize=(16, 8))
ax1.plot(train['id'], train['loss'])
ax1.set_title('Loss values per id')
ax1.set_xlabel('id')
ax1.set_ylabel('loss')
ax1.legend()
plt.show()

#数据有倾斜，这样的数据分布使得这个功能非常扭曲导致的回归表现不佳。但为了评价实际偏差，使用偏度
print(stats.mstats.skew(train['loss'])) #右偏。比1大
#使用log转换
print(stats.mstats.skew(np.log(train['loss']))) #0.09
#看下变化后的对比
fig2, ax2 = plt.subplots(1,2, figsize=(16, 5))
ax2[0].hist(train['loss'], bins=50)
ax2[0].set_title('Train Loss target histogram')
ax2[0].grid(True)
ax2[1].hist(np.log(train['loss']), bins=50, color='g')
ax2[1].set_title('Train Log Loss target histogram')
ax2[1].grid(True)
plt.show()


#可视化各连续值的分布
train[cont_features].hist(bins=50, figsize=(16, 12))
plt.show()

#可视化数值型特征之间的相关性
fig3, ax3 = plt.subplots(figsize=(16,9))
correation = train[cont_features].corr()
print(correation)
sns.heatmap(correation, annot=True, ax=ax3)
plt.show()
#几个特征之间有很高的相关性，相关太高，存在共线性剔除。进行筛选
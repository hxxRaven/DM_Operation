import sys
import pickle
import pandas as pd
import numpy as np
import os
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

#分类函数
def threshold(y, t):
    z = np.copy(y)
    z[z>=t] = 1
    z[z<t] = 0
    return z


data_path = 'data'
df_path = 'df293.pkl'
text_path = 'text_features_1.pkl'
words_path ='single_select_words.pkl'


df = pickle.load(open(data_path+os.sep+df_path, 'rb'))
text = pickle.load(open(data_path+os.sep+text_path, 'rb'))
df = df.merge(text, on='CUST_NO', how='left')

#切分训练测试
train = df.loc[df.label != -1]
test = df.loc[df.label == -1]

print('训练集：',train.shape[0])
print('正样本:',train.loc[train.label == 1].shape[0])
print('负样本:',train.loc[train.label == 0].shape[0])
print('测试集：',test.shape[0])


x_data = train.copy()
x_val = test.copy()
x_data = x_data.sample(frac=1, random_state=1).reset_index(drop=True)   #洗牌

#去除即将要处理的特征，稀疏矩阵处理
delete_columns = ['CUST_NO', 'label', 'contents']
X_train_1 = csc_matrix(x_data.drop(delete_columns, axis=1).values)
X_val_1 = csc_matrix(x_val.drop(delete_columns, axis=1).values())
y_train = x_data.label.values
y_val = x_val.label.values

#上一步的选择好的词用来做tfidf
print('tfidf...')
featurenames = list(x_data.drop(delete_columns, axis=1).columns)
select_words = pickle.load(open(data_path+os.sep+words_path, 'rb'))
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, use_idf=False, smooth_idf=False, sublinear_tf=True, vocabulary=select_words)
tfidf.fit(x_data['contents'])
word_names = tfidf.get_feature_names()
X_train_2 = tfidf.transform(x_data['contents'])
X_val_2 = tfidf.transform(x_val['contents'])

print('文本特征：{}维'.format(len(word_names)))
statistic_feature = featurenames.copy()
print('其他特征：{}维'.format(len(statistic_feature)))
featurenames.extend(word_names)


from scipy.sparse import hstack
X_train = hstack(((X_train_1), (X_train_2))).tocsc()    #去掉contents和tfidf之后的contents合并再稀疏矩阵
X_val = hstack(((X_val_1), (X_val_2))).tocsc()

print('特征数量',X_train.shape[1])


#开始针对种子数不同来训练模型，并bagging结果
import time
print('start 3 xgboost!')
st = time.perf_counter()
bagging = []
for i in range(1,4):
    print('group:',i)


    print('training...')
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=featurenames)
    dval = xgb.DMatrix(X_val, feature_names=featurenames)
    params = {"objevtive":"binary:logistic",
              "booster": "gbtree",
              "eval_metric": "error",
              "eta": 0.1,
              'max_depth':14,
              'subsample':0.8,
              'min_child_weight':2,
              'colsample_bytree':1,
              'gamma':0.2,
              "lambda":300,
              'silent':1,
              "seed":i,
              }
    watchlist = [(dtrain, 'train')]

    model = xgb.train(params, dtrain, 2000, evals=watchlist, early_stopping_rounds=50, verbose_eval=100)
    print('Predicting......')
    y_prob = model.predict(dval, ntree_limit=model.best_ntree_limit)
    bagging.append(y_prob)


print('String voting.....')
t = 0.5
pres = []
for i in bagging:
    pres.append(threshold(i, t))
print(pres)

pres = np.array(pres).T.astype('int64') #转置才能构成一行三个预测结果那种情况
result = []
for line in pres:
    result.append(np.bincount(line).argmax()) #取出现次数最多的

#构建输出结果
myout = test[['CUST_NO']].copy()
myout['pre'] = result
print('finished')
myout.loc[myout['pre'] == 1, 'CUST_NO'].to_csv('results'+os.sep+'result.csv', index=False)
et = time.perf_counter()
print(et-st)
'''
文本特征不一定越多越好，达到一定程度会干扰。
文本特征一般维度都较高，之前处理各种编号就是减少维度
特征重要性，建立模型，无效化某特征，看正确性
'''

#选择特征
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb #https://www.lfd.uci.edu/~gohlke/pythonlibs/
import os
import pickle

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)



data_path = 'data'
df_path = 'df293.pkl'
text_path = 'text_features_1.pkl'
jobinfo_path = 'job294.pkl'

df = pickle.load(open(data_path+os.sep+df_path, 'rb'))
text = pickle.load(open(data_path+os.sep+text_path, 'rb'))
jobinfo = pickle.load(open(data_path+os.sep+jobinfo_path, 'rb'))
print(df.shape)
print(df.columns)
#把text合入df
df = df.merge(text, on='CUST_NO', how='left')
print(df.shape)
print(df.columns)
train = df.loc[df['label'] != -1]
test = df.loc[df['label'] == -1]

x_data = train.copy()
x_val = train.copy()

#洗牌训练集
x_data = x_data.sample(frac=1, random_state =1).reset_index(drop=True)
print(type(x_data))
print(type(x_val))
#删除不用的特征
delete_columns = ['CUST_NO', 'label', 'contents']

#因为onehot编码导致很多列都是零，所以用稀疏矩阵csc_matrix记住那些非零位置的索引
X_train_1 = csc_matrix(x_data.drop(delete_columns, axis=1).values)
X_val_1 = csc_matrix(x_val.drop(delete_columns, axis=1).values)

y_train = x_data['label'].values
y_val = x_val['label'].values
# print(X_train_1)
'''
TF_IDF
A词在大型语料库中出现的很多，在我们要分析的文章里出现了10次
B词在大型语料库中出现的很少，在我们要分析的文章里出现了10次
B词更能代表我们要分析文章的特点。TD-IDF大
'''
print('select features...')
features_names =list(x_data.drop(delete_columns, axis=1).columns)
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, use_idf=False, smooth_idf=False, sublinear_tf=True)
tfidf.fit(x_data['contents'])
word_name = tfidf.get_feature_names()
X_train_2 = tfidf.transform(x_data['contents']) #特征已经转为tfidf后的特征
print ('X_train_2[1]:',X_train_2[1])    #看下哪些位置有值，其余都是空
X_val_2 = tfidf.transform(x_val['contents'])
print('文本特征：{}维'.format(len(word_name)))
static_features = features_names.copy()
print('其他特征：{}维'.format(len(static_features)))
features_names.extend(word_name)    #合并
#tfidf为了保证词汇都被计算到，所以使用了很全的词汇作为特征标记，这样就又导致了稀疏矩阵
from scipy.sparse import hstack
X_train = hstack(((X_train_1),(X_train_2))).tocsc()   #X_train_1去除了contents，X_train_2是tfidf之后的contents
X_val = hstack(((X_val_1),(X_val_2))).tocsc()

print('特征数量',X_train.shape[1])


'''特征太多了，要进行降维，这里使用xgboost筛选特征'''
print('training...')
dtrain = xgb.DMatrix(X_train, y_train, feature_names=features_names)
dval = xgb.DMatrix(X_val, feature_names=features_names)

params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "error",
    "eta": 0.1,
    'max_depth':12,
    'subsample':0.8,
    'min_child_weight':3,
    'colsample_bytree':1,
    'gamma':0.2,
    "lambda":300,
    "silent":1,
    'seed':1,
}
watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, 2500, evals=watchlist, early_stopping_rounds=100, verbose_eval=100)
print('finished')
temp = pd.DataFrame.from_dict(model.get_fscore(), orient='index').reset_index() #取得每个特征得分值
temp.columns = ['features','score']
temp.sort_values(['score'], axis=0, ascending=False, inplace=True)
temp = temp.reset_index(drop=True)

print('留下文本特征数量：', len(temp.loc[~temp.features.isin(static_features)]))
#选择出那些降维后得分值高的特征名，所以排除掉那些在x_data原始数据中出现的词
selected_words = list(temp.loc[~temp.feature.isin(static_features)].feature.values)

pickle.dump(selected_words, open(data_path+os.sep+'single_select_words.pkl', 'wb'))
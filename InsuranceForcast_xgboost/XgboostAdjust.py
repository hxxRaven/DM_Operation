import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys, os, time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

'''Xgboost调参策略'''

#在大部分数据已经调整完毕的情况下，对右偏的loss作为label进行log变换
data_path = 'data' + os.sep + 'train.csv'
train = pd.read_csv(data_path)
train['log_loss'] = np.log(train['loss'])

#提取特征
print(train.info())
features = [x for x in train.columns if x not in ['id', 'loss', 'log_loss']]
cat_features = [x for x in train[features].select_dtypes(include=['object']).columns]   #字符型特征
num_features = [x for x in train[features].select_dtypes(exclude=['object']).columns]   #非字符型特征
print ("Categorical features:", len(cat_features))
print ("Numerical features:", len(num_features))

rtrain = train.shape[0]
train_x = train[features]
train_y = train['log_loss']

#把这些二值化多的参数映射到category类型
for c in range(len(cat_features)):
    train_x[cat_features[c]] =  train_x[cat_features[c]].astype('category').cat.codes
print(train_x[cat_features])
print ("Xtrain:", train_x.shape)
print ("ytrain:", train_y.shape)


'''
首先，我们训练一个基本的xgboost模型，然后进行参数调节通过交叉验证来观察结果的变换，使用平均绝对误差来衡量
mean_absolute_error(np.exp(y), np.exp(yhat))。
'''

#xgboost 自定义了一个数据矩阵类 DMatrix，会在训练开始时进行一遍预处理，从而提高之后每次迭代的效率
dtrain = xgb.DMatrix(train_x, train_y)  #先特征， 后label

#评估函数，结果看起来幂级增长
def xg_eval_mae(y_for, rtrain):
    y = rtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(y_for))


'''
Xgboost参数
'booster':'gbtree',
'objective': 'multi:softmax', 多分类的问题
'num_class':10, 类别数，与 multisoftmax 并用
'gamma':损失下降多少才进行分裂
'max_depth':12, 构建树的深度，越大越容易过拟合
'lambda':2, 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, 随机采样训练样本
'colsample_bytree':0.7, 生成树时进行的列采样
'min_child_weight':3, 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
'silent':0 ,设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, 如同学习率
'seed':1000,
'nthread':7, cpu 线程数
'''

#基础模型定义
xgb_params = {'seed':0,
              'eta':0.1,
              'colsample_bytree':0.5,
              'silent':1,
              'subsample':0.5,
              'objective':'reg:linear',
              'max_depth':5,
              'min_child_weight':3}
#交叉验证,num_boost_round树的个数，feval指定评价函数,maximize是否对评估函数进行最大化,
#early_stopping_rounds,早期停止次数,如果early_stopping_rounds 存在，则模型会生成三个属性，bst.best_score,bst.best_iteration,和bst.best_ntree_limit
start_time = time.perf_counter()
bst_cvl = xgb.cv(xgb_params, dtrain, num_boost_round=50,
                 nfold=3, seed=0, feval=xg_eval_mae,
                 maximize=False, early_stopping_rounds=10)
end_time = time.perf_counter()
#一开始让结果大，方便对比
print('CV score:', bst_cvl.iloc[-1,:]['test-mae-mean'])
print(end_time-start_time)

plt.figure()
bst_cvl[['train-mae-mean','test-mae-mean']].plot()  #看下效果，随着迭代进行，loss下降趋于平稳接近饱和。
plt.show()
##我们的第一个基础模型：1.没有发生过拟合 2.只建立了50个树模型##

#试下100课树
start_time = time.perf_counter()
bst_cv2 = xgb.cv(xgb_params, dtrain, num_boost_round=100,
                 nfold=3, seed=0, feval=xg_eval_mae, maximize=False,
                 early_stopping_rounds=10)
end_time = time.perf_counter()
print ('CV score:', bst_cv2.iloc[-1,:]['test-mae-mean'])    #效果会好些
print(end_time-start_time)
#可视化展示
fig1, ax1 = plt.subplots(1,2, figsize=(16, 4))
ax1[0].set_title('100 rounds of training')
ax1[0].set_xlabel('Rounds')
ax1[0].set_ylabel('Loss')
ax1[0].grid(True)
ax1[0].plot(bst_cv2[['train-mae-mean', 'test-mae-mean']])
ax1[0].legend(['Training Loss', 'Test Loss'])
#因为1图的坐标轴太大，y轴过大，x轴40之后的变化很细微。所以2图只显示x40之后的
ax1[1].set_title('60 last rounds of training')
ax1[1].set_xlabel('Rounds')
ax1[1].set_ylabel('Loss')
ax1[1].grid(True)
ax1[1].plot(bst_cv2.iloc[40:][['train-mae-mean', 'test-mae-mean']])
ax1[1].legend(['Training Loss', 'Test Loss'])
#有那么一丁丁过拟合，现在还没多大事，但饱和趋势放缓并没停止

# fig, (ax1, ax2) = plt.subplots(1,2)
# fig.set_size_inches(16,4)
# ax1.set_title('100 rounds of training')
# ax1.set_xlabel('Rounds')
# ax1.set_ylabel('Loss')
# ax1.grid(True)
# ax1.plot(bst_cv2[['train-mae-mean', 'test-mae-mean']])
# ax1.legend(['Training Loss', 'Test Loss'])
# ax2.set_title('60 last rounds of training')
# ax2.set_xlabel('Rounds')
# ax2.set_ylabel('Loss')
# ax2.grid(True)
# ax2.plot(bst_cv2.iloc[40:][['train-mae-mean', 'test-mae-mean']])
# ax2.legend(['Training Loss', 'Test Loss'])

'''
调节参数
Step 1: 选择一组初始参数
Step 2: 改变 max_depth 和 min_child_weight.
Step 3: 调节 gamma 降低模型过拟合风险.
Step 4: 调节 subsample 和 colsample_bytree 改变数据采样策略.
Step 5: 调节学习率 eta.
'''
class XGBoostRegressor(object):
    def __init__(self, **kwargs):   #默认参数
        self.params = kwargs
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'silent': 1, 'objective': 'reg:linear', 'seed': 0})

    def fit(self, x_train, y_train):    #建立初始树，并查看效果，收集指标
        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                             feval=xg_eval_mae, maximize=False)

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)

    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)
        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=nfold, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
        return cv_rounds.iloc[-1:]

    def plot_feature_importances(self): #可视化
        feat_imp = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self


def mae_score(y_true, y_pred):  #评分
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))


#开始

#初始模型
bst = XGBoostRegressor(eta=0.1, colsample_bytree=0.5, subsample=0.5,
                       max_depth=5, min_child_weight=3, num_boost_round=50)
print(bst.kfold(train_x, train_y, nfold=5).T)

'''树个数之前已经完成'''

'''树的深度与节点权重
max_depth: 树的最大深度。增加这个值会使模型更加复杂，也容易出现过拟合，深度3-10是合理的。
min_child_weight: 正则化参数. 如果树分区中的实例权重小于定义的总和，则停止树构建过程。
'''
xgb_param_grid = {'max_depth':list(range(4,9)), 'min_child_weight':list(1,3,6)}
print(xgb_param_grid['min_child_weight'])

start_time = time.perf_counter()
grid = GridSearchCV(XGBRegressor(eta=0.1, num_boost_round=50, colsample_bytree=0.5, subsample=0.5),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_score)
grid.fit(train_x, train_y.values)
print(grid.cv_results_['mean_test_score'],grid.cv_results_['params'], grid.best_params_, grid.best_score_)
end_time = time.perf_counter()
print(end_time-start_time)
#输出打分值（mean不看正负，只看和零的差值）
#可视化网格搜索发现的最佳结果:{'max_depth': 8, 'min_child_weight': 6},-1187.9597499123447)

'''Step 3: 调节 gamma去降低过拟合风险'''

xgb_param_grid = {'gamma':[ 0.1 * i for i in range(0,5)]}
start_time = time.perf_counter()
grid = GridSearchCV(XGBoostRegressor(eta=0.1, num_boost_round=50, max_depth=8, min_child_weight=6,
                                     colsample_bytree=0.5, subsample=0.5),
                                     param_grid=xgb_param_grid, cv=5, scoring=mae_score)
grid.fit(train_x, train_y.values)
end_time = time.perf_counter()
print(grid.cv_results_['mean_test_score'],grid.cv_results_['params'], grid.best_params_, grid.best_score_)
print(end_time-start_time)

'''Step 4: 调节样本采样方式 subsample 和 colsample_bytree'''
xgb_param_grid = {'subsample':[ 0.1 * i for i in range(6,9)],
                  'colsample_bytree':[ 0.1 * i for i in range(6,9)]}
grid = GridSearchCV(XGBoostRegressor(eta=0.1, gamma=0.2, num_boost_round=50, max_depth=8, min_child_weight=6),
                                     param_grid=xgb_param_grid, cv=5, scoring=mae_score)
start_time = time.perf_counter()
grid.fit(train_x, train_y.values)
end_time = time.perf_counter()
print(grid.cv_results_['mean_test_score'],grid.cv_results_['params'], grid.best_params_, grid.best_score_)
print(end_time-start_time)


'''Step 5: 减小学习率并增大树个数,一般减小学习率增大树的个数'''
xgb_param_grid = {'eta':[0.5,0.4,0.3,0.2,0.1,0.075,0.05,0.04,0.03]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=50, gamma=0.2, max_depth=8, min_child_weight=6,
                                     colsample_bytree=0.6, subsample=0.9),
                                    param_grid=xgb_param_grid, cv=5, scoring=mae_score)
start_time = time.perf_counter()
grid.fit(train_x, train_y.values)
end_time = time.perf_counter()
print(grid.cv_results_['mean_test_score'],grid.cv_results_['params'], grid.best_params_, grid.best_score_)
print(end_time-start_time)
#此时学习率0.2最优，1160

'''学习率和树的关系值得探究，再试试100棵树'''
xgb_param_grid = {'eta':[0.5,0.4,0.3,0.2,0.1,0.075,0.05,0.04,0.03]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=100, gamma=0.2, max_depth=8, min_child_weight=6,
                                     colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_score)

start_time = time.perf_counter()
grid.fit(train_x, train_y.values)
end_time = time.perf_counter()
print(grid.cv_results_['mean_test_score'],grid.cv_results_['params'], grid.best_params_, grid.best_score_)
print(end_time-start_time)
#学习率0.1最优，1152

'''再试200棵树'''
xgb_param_grid = {'eta':[0.09,0.08,0.07,0.06,0.05,0.04]}
grid = GridSearchCV(XGBoostRegressor(num_boost_round=200, gamma=0.2, max_depth=8, min_child_weight=6,
                                     colsample_bytree=0.6, subsample=0.9),
                    param_grid=xgb_param_grid, cv=5, scoring=mae_score)
start_time = time.perf_counter()
grid.fit(train_x, train_y.values)
end_time = time.perf_counter()
print(grid.cv_results_['mean_test_score'],grid.cv_results_['params'], grid.best_params_, grid.best_score_)
print(end_time-start_time)
#学习率0.07最优，1145
''' 
100 trees, eta=0.1: MAE=1152.247
200 trees, eta=0.07: MAE=1145.92
学习率和树的关系并非线性，但相关
XGBoostRegressor(num_boost_round=200, gamma=0.2, max_depth=8, min_child_weight=6, colsample_bytree=0.6, subsample=0.9, eta=0.07).
'''
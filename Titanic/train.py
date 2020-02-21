import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

import os
path = 'data' + os.sep + 'Features1.csv'
data =pd.read_csv(path, index_col=0)

print(data.head())

'''
机器学习建模
我们从EDA部分获得了一些见解。但是，我们不能准确地预测或判断一个乘客是否会幸存或死亡。现在我们将使用一些很好的分类算法来预测乘客是否能生存下来：
1）logistic回归
2）支持向量机（线性和径向）
3）随机森林
4）k-近邻
5）朴素贝叶斯
6）决策树
7）神经网络
'''
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

'''权重参数，'''
from sklearn.model_selection import train_test_split
from sklearn import metrics #评估
from sklearn.metrics import confusion_matrix #混淆矩阵评估

'''切分训练测试集'''
train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Survived'])
train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]
X = data[data.columns[1:]]
Y = data['Survived']


'''支持向量机'''
#rbf
model_svm1 = svm.SVC(kernel='rbf', C=1, gamma=0.1)   #核变化函数，软间隔力度，变化复杂程度
model_svm1.fit(train_X, train_Y)
prediction_svm1 = model_svm1.predict(test_X)
# print(prediction_svm1)
# print(test_Y)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction_svm1,test_Y))

#linear
model_svm2 = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
model_svm2.fit(train_X, train_Y)
prediction_svm2 = model_svm2.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction_svm2,test_Y))


'''逻辑回归'''
model_lr = LogisticRegression()
model_lr.fit(train_X,train_Y)
prediction_lr=model_lr.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction_lr,test_Y))


'''决策树'''
model_dt=DecisionTreeClassifier()
model_dt.fit(train_X,train_Y)
prediction_dt=model_dt.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction_dt,test_Y))


'''KNN'''
model_knn=KNeighborsClassifier()
model_knn.fit(train_X,train_Y)
prediction_knn=model_knn.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction_knn,test_Y))
#针对KNN的参数N，可以设置下遍历看下效果
a_index = list(range(1, 11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in a_index:
    mode_knnn = KNeighborsClassifier(n_neighbors=i)
    mode_knnn.fit(train_X, train_Y)
    pred_knnn = mode_knnn.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(pred_knnn, test_Y)))

fig1, ax1 = plt.subplots()
ax1.plot(a_index, a)
plt.xticks(x)
fig1.set_size_inches(12,6)
plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())

# a_index=list(range(1,11))
# a=pd.Series()
# x=[0,1,2,3,4,5,6,7,8,9,10]
# for i in a_index:
#     model=KNeighborsClassifier(n_neighbors=i)
#     model.fit(train_X,train_Y)
#     prediction=model.predict(test_X)
#     a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
# plt.plot(a_index, a)
# plt.xticks(x)
# fig=plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()
# print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())


'''贝叶斯'''
model_bayes = GaussianNB()
model_bayes.fit(train_X, train_Y)
prediction_bayes = model_bayes.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction_bayes,test_Y))


'''随机森林'''
model_rf=RandomForestClassifier(n_estimators=100)
model_rf.fit(train_X,train_Y)
prediction_rf=model_rf.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction_rf,test_Y))
##现在这个分类器的精确度很高，但是我们可以确认所有的新测试集都是90%吗？答案是否定的，因为我们不能确定分类器在不同数据源上的结果。当训练和测试数据发生变化时，精确度也会改变。它可能会增加或减少。##
##为了克服这一点，得到一个广义模型，我们使用交叉验证。##


'''
交叉验证
一个测试集看起来不太够呀，多轮求均值是一个好的策略！
1）的交叉验证的工作原理是首先将数据集分成k-subsets。
2）假设我们将数据集划分为（k＝5）部分。我们预留1个部分进行测试，并对这4个部分进行训练。
3）我们通过在每次迭代中改变测试部分并在其他部分中训练算法来继续这个过程。然后对衡量结果求平均值，得到算法的平均精度。
'''

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kf = KFold(n_splits=10, random_state=22)
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
cv_results = pd.DataFrame({'CV mean':xyz, 'CV std': std}, index=classifiers)
print(cv_results)

plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()


'''混淆矩阵的热度图表示'''
fig3, ax3=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax3[0,0],annot=True,fmt='2.0f')
ax3[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax3[0,1],annot=True,fmt='2.0f')
ax3[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax3[0,2],annot=True,fmt='2.0f')
ax3[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax3[1,0],annot=True,fmt='2.0f')
ax3[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax3[1,1],annot=True,fmt='2.0f')
ax3[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax3[1,2],annot=True,fmt='2.0f')
ax3[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax3[2,0],annot=True,fmt='2.0f')
ax3[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()
#横坐标为预测，纵坐标为实际
##预测的正确率为491（死亡）+ 247（存活），平均CV准确率为（491+247）/ 891＝82.8%。recall也可以##



'''
超参数选择
机器学习模型就像一个黑盒子。这个黑盒有一些默认参数值，我们可以调整或更改以获得更好的模型。比如支持向量机模型中的C和γ，我们称之为超参数，他们对结果可能产生非常大的影响。
'''
#网格搜索GridSearch
from sklearn.model_selection import GridSearchCV

#SVM
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hypers_svm = {'kernel':kernel, 'C':C, 'gamma':gamma}
gs_svm = GridSearchCV(estimator=svm.SVC(), param_grid=hypers_svm, verbose=True)
gs_svm.fit(X, Y)
print(gs_svm.best_score_)
print(gs_svm.best_params_)

#随机森林
n_estimators=range(100,1000,100)
hypers_rf={'n_estimators':n_estimators}
gs_rf=GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=hypers_rf,verbose=True)
gs_rf.fit(X,Y)
print(gs_rf.best_score_)
print(gs_rf.best_estimator_)


'''
集成
1）随机森林类型或者Bagging的，并行的集成
2）提升类型,Xgboost
3）堆叠类型(第一阶段用什么分类器得到结果输入到第二阶段分类器)
'''
#投票分类器
#这是将许多不同的简单机器学习模型的预测结合起来的最简单方法。它给出了一个平均预测结果基于各子模型的预测。
from sklearn.ensemble import VotingClassifier
ensemble_voting = VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], voting='soft').fit(train_X, train_Y)
print('The accuracy for ensembled model is:',ensemble_voting.score(test_X,test_Y))  #VotingClassifier对象只有score没有accuary——score
cross=cross_val_score(ensemble_voting, X, Y, cv = kf, scoring = "accuracy")
print('The cross validated score is', cross.mean())

#Bagging
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700) #n_estimators同时建立的树
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction, test_Y))
result=cross_val_score(model, X, Y, cv=kf, scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())

#Bagging DecisionTree
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())

#AdaBoost提升模型
#升是一个逐步增强的弱模型：
# 首先对完整的数据集进行训练。现在模型会得到一些实例，而有些错误。现在，在下一次迭代中，学习者将更多地关注错误预测的实例或赋予它更多的权重
# AdaBoost（自适应增强） 在这种情况下，弱学习或估计是一个决策树。但我们可以改变缺省base_estimator任何算法的选择。
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
#GridSearch选择参数
n_estimators=list(range(100, 1100, 100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hypers_adb = {'n_estimators':n_estimators, 'learning_rate':learn_rate}
gs_adb = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=hypers_adb, verbose=True)
gs_adb.fit(X, Y)
print(gs_adb.best_params_)
print(gs_adb.best_score_)
##我们可以从AdaBoost的最高精度是83.16%，n_estimators = 200和learning_rate = 0.05##
#用此数据展示下混淆矩阵
adb = AdaBoostClassifier(n_estimators=200, random_state=0, learn_rate=0.05)
result_adb = cross_val_predict(adb, X, Y, cv=kf)
sns.heatmap(confusion_matrix(Y, result_adb), cmap='winter', annot=True, fmt='2.0f')
plt.show()

#GradientBoost提升模型
from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())


'''特征的重要程度'''
fig4, ax4 = plt.subplots(2, 2, figsize=(15, 12))

model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax4[0,0])
ax4[0,0].set_title('Feature Importance in Random Forests')

model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax4[0,1],color='#ddff11')
ax4[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax4[1,0],cmap='RdYlGn_r')
ax4[1,0].set_title('Feature Importance in Gradient Boosting')

import xgboost as xg
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax4[1,1],color='#FD0F00')
ax4[1,1].set_title('Feature Importance in XgBoost')

plt.show()
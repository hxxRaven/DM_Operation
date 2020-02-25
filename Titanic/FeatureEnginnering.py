import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

import os
path = 'InsuranceForcast_xgboost.data' + os.sep + 'preprocess1.csv'
data =pd.read_csv(path)
print(data.head())

'''
特征之间的相关性，如果太过相关则共性，可以舍弃其中之一
特征相关性的热度图
首先要注意的是，只有数值特征进行比较
正相关：如果特征A的增加导致特征b的增加，那么它们呈正相关。值1表示完全正相关。
负相关：如果特征A的增加导致特征b的减少，则呈负相关。值-1表示完全负相关。
现在让我们说两个特性是高度或完全相关的，所以一个增加导致另一个增加。这意味着两个特征都包含高度相似的信息，并且信息很少或没有变化。这样的特征对我们来说是没有价值的！
那么你认为我们应该同时使用它们吗？。在制作或训练模型时，我们应该尽量减少冗余特性，因为它减少了训练时间和许多优点。
现在，从上面的图，我们可以看到，特征不显著相关。
'''
# sns.heatmap(InsuranceForcast_xgboost.data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2)
# fig = plt.gcf()
# fig.set_size_inches(10, 8)
# plt.show()
fig1, ax1 = plt.subplots(figsize=(15, 16))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2, ax=ax1)
fig1 = plt.gcf()
plt.show()

'''
特征工程和数据清洗
当我们得到一个具有特征的数据集时，是不是所有的特性都很重要？可能有许多冗余的特征应该被消除，我们还可以通过观察或从其他特征中提取信息来获得或添加新特性。
'''



'''年龄特征(离散化)
正如我前面提到的，年龄是连续的特征，在机器学习模型中存在连续变量的问题。
如果我说通过性别来组织或安排体育运动，我们可以很容易地把他们分成男女分开。
如果我说按他们的年龄分组，你会怎么做？如果有30个人，可能有30个年龄值。
我们需要对连续值进行离散化来分组。
好的，乘客的最大年龄是80岁。所以我们将范围从0-80成5箱。所以80/5＝16。'''
data['Age_band'] = 0
data.loc[data['Age']<=16, 'Age_band'] = 0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
print(data.head())
print(data['Age_band'].value_counts())

# fig, ax = plt.subplots(figsize=(12,8))
sns.factorplot('Age_band', 'Survived', data=data, col='Pclass')
plt.show()
#对结果的影响很大，可以加入考虑


'''
Family_size：家庭总人数
全家的人数
'''
data['Family_Size'] = 0
data['Family_Size'] = data['Parch'] + data['SibSp']
data['Alone'] = 0   #是否独身一人
data.loc[data['Family_Size']==0, 'Alone'] = 1

fig2, ax2 = plt.subplots(1, 2, figsize=(18, 7))
sns.barplot('Family_Size', 'Survived', data=data, ax=ax2[0])
ax2[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=data,ax=ax2[1])
ax2[1].set_title('Alone vs Survived')
plt.show()
##如果你是单独或family_size = 0，那么生存的机会很低。家庭规模4以上，机会也减少。这看起来也是模型的一个重要特性。让我们进一步研究这个问题。##

sns.factorplot('Alone','Survived',data=data,hue='Sex',col='Pclass')#看下独身，生存，性别和船舱的生存
plt.show()

'''
船票价格的离散化
'''
data['Fare_Range'] = pd.qcut(data['Fare'], 4)
print(pd.crosstab(data['Fare_Range'], data['Survived'] ))

#船票价格增加生存的机会增加。
data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

sns.factorplot('Fare_cat','Survived',data=data,hue='Sex')   #绘图我们再加上颜色试试看
plt.show()
##随着fare_cat增加，存活的几率增加。随着性别的变化，这一特性可能成为建模过程中的一个重要特征。##


'''
将字符串值转换为数字 因为我们不能把字符串一个机器学习模型
这些映射关系需要做记录的
'''
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

'''
去掉不必要的特征
名称>我们不需要name特性，因为它不能转换成任何分类值
年龄——>我们有age_band特征，所以不需要这个
票号-->这是任意的字符串，不能被归类
票价——>我们有fare_cat特征，所以不需要
船仓号——>这个也不要没啥含义
passengerid -->不能被归类
'''

data.drop(['Unnamed: 0', 'Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis=1, inplace=True)
fig3, ax3 = plt.subplots(figsize=(12,9))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20}, ax=ax3)
fig3.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

data.to_csv('InsuranceForcast_xgboost.data' + os.sep + 'Features1.csv')
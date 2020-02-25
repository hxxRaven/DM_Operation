'''
数据挖掘流程：
（一）数据读取：

读取数据，并进行展示
统计数据各项指标
明确数据规模与要完成任务
（二）特征理解分析

单特征分析，逐个变量分析其对结果的影响
多变量统计分析，综合考虑多种情况影响
统计绘图得出结论
（三）数据清洗与预处理

对缺失值进行填充
特征标准化/归一化
筛选有价值的特征
分析特征之间的相关性
（四）建立模型

特征数据与标签准备
数据集切分
多种建模算法对比
集成策略等方案改进
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

import os
path = 'InsuranceForcast_xgboost.data' + os.sep + 'train.csv'

data =pd.read_csv(path)
# print(InsuranceForcast_xgboost.data.head())


print(data.isnull().sum())  #查看缺失值

print(data.info())  #查看数据类型和分布
print(data.describe())

#查看生存的比例数据
# print(InsuranceForcast_xgboost.data['Survived'].value_counts())
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived', data=data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()
##仅38%的人获救了，下面探讨可能因素##

'''
数据特征分为：连续值和离散值
离散值：性别（男，女） 登船地点（S,Q,C）
连续值：年龄，船票价格
'''

'''先看性别对于的生存率的影响'''
print(data.groupby(['Sex','Survived'])['Survived'].count()) #先分性别，再分生死，填入分好后Survived的计数
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax1[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=data, ax=ax1[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
##男的很多，但最后生存率女的多，而且女的生存总数也比男的多，所以性别这个指标很重要#

'''再看看船舱等级的影响'''
print(pd.crosstab(data['Pclass'], data['Survived'], margins=True))

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax2[0])
ax2[0].set_title('Number Of Passengers By Pclass')
ax2[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=data, ax=ax2[1])
ax2[1].set_title('Pclass:Survived vs Dead')
plt.show()
##一等舱人数最少，但活的最多，三等 舱人数最多，但获救比例远小于一等舱，也小于二等舱，所以pclass也很重要##

'''那么在不同等级船舱中，性别对生存率是否有影响呢'''
print(pd.crosstab([data['Sex'], data['Survived']], data['Pclass'], margins=True))
sns.factorplot('Pclass','Survived',hue='Sex',data=data)
plt.show()
##男女生存率都呈现下降趋势，但女性确实较高##


'''再看下年龄这个连续值对结果的影响'''
#violinplot展示三维数据，1横坐标，2纵坐标， 3坐标再二分类一左一右，表示变化
fig3, ax3 = plt.subplots(1, 2, figsize=(12,7))
sns.violinplot('Pclass', 'Age', hue='Survived', data=data, split=True, ax=ax3[0])
ax3[0].set_title('Pclass and Age vs Survived')
ax3[0].set_yticks(range(0,110,10))
sns.violinplot('Sex', 'Age', hue='Survived', data=data, split=True, ax=ax3[1])
ax3[1].set_title('Sex and Age vs Survived')
ax3[1].set_yticks(range(0,110,10))
plt.show()
##1）10岁以下儿童的存活率随pclass数量增加。2）生存为20-50岁获救几率更高一些。3）对男性来说，随着年龄的增长，存活率降低。##

'''
缺失值填充
平均值（中位数）
经验值
回归模型预测
剔除掉 3/10的缺失

正如我们前面看到的，年龄特征有177个空值。为了替换这些缺失值，我们可以给它们分配数据集的平均年龄。
但问题是，有许多不同年龄的人。最好的办法是找到一个合适的年龄段！
我们可以检查名字特征。根据这个特征，我们可以看到名字有像先生或夫人这样的称呼，这样我们就可以把先生和夫人的平均值分配给各自的组。
'''
#1.检查名字特征，按Mrs、Miss和Mrs填充期间值
data['Initial'] = 0
for i in data:
    data['Initial'] = data['Name'].str.extract('([A-Za-z]+)\.') #正则提取

print(data['Initial'])
print(pd.crosstab(data.Sex,data.Initial)) #看下性别和称呼的分布

data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

print(data.groupby('Initial')['Age'].mean())

data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Mr'), 'Age'] = 33
data.loc[(data.Age.isnull()) & (data.Initial == 'Mrs'),'Age'] = 36
data.loc[(data.Age.isnull()) & (data.Initial == 'Master'),'Age']= 5
data.loc[(data.Age.isnull()) & (data.Initial == 'Miss'),'Age'] = 22
data.loc[(data.Age.isnull()) & (data.Initial == 'Other'),'Age'] = 46
print(data['Age'].isnull().sum())   #查看是否还有空值
#看下年龄和获救的关系，hist图
fig4, ax4=plt.subplots(1,2,figsize=(20,10))
data[data['Survived']==0].Age.plot.hist(ax=ax4[0],bins=20,edgecolor='black',color='red')
ax4[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax4[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax4[1],color='green',bins=20,edgecolor='black')
ax4[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax4[1].set_xticks(x2)
plt.show()
##1）幼儿（年龄在5岁以下）获救的还是蛮多的（妇女和儿童优先政策）。2）最老的乘客得救了（80年）。3）死亡人数最高的是30-40岁年龄组。##

sns.factorplot('Pclass', 'Survived', hue='Initial', data=data)
plt.show()

'''
Embarked--> 登船地点
'''
#看下登船点，船舱等级，性别和生存率的关系
print(pd.crosstab([data['Embarked'], data['Pclass']], [data['Sex'], data['Survived']], margins=True))
sns.factorplot('Embarked', 'Survived', data=data)
plt.show()
#C港获救人数最高，再看下分开情况下每个特征和Embarked的关系
fig5, ax5 = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked',data=data,ax=ax5[0,0])
ax5[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax5[0,1])
ax5[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax5[1,0])
ax5[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax5[1,1])
ax5[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
#1）大部分人的船舱等级是3。2）C的乘客看起来很幸运，他们中的一部分幸存下来。3）S港口的富人蛮多的。仍然生存的机会很低。4）港口Q几乎有95%的乘客都是穷人。

sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)  #col=将整图分为三列，以Embarked为列
plt.show()
##观察:1）存活的几率几乎为1 在pclass1和pclass2中的女人。2）pclass3 的乘客中男性和女性的生存率都是很偏低的。3）端口Q很不幸，因为那里都是3等舱的乘客。##

#港口中也存在缺失值，在这里我用众数来进行填充了，因为S登船人最多呀
# InsuranceForcast_xgboost.data.loc[InsuranceForcast_xgboost.data['Embarked'].isnull(), 'Embarked'] = 'S'
data['Embarked'].fillna('S',inplace=True)
print(data['Embarked'].isnull().sum())


'''不同兄弟姐妹数量度结果的影响'''
print(pd.crosstab(data['SibSp'], data['Survived']))
fig6, ax6 = plt.subplots(figsize=(12,7))
sns.barplot('SibSp', 'Survived', data=data, ax=ax6)
ax6.set_title('SibSp vs Survived')
plt.show()
##如果乘客是孤独的船上没有兄弟姐妹，他有34.5%的存活率。如果兄弟姐妹的数量增加，该图大致减少。这是有道理的。也就是说，如果我有一个家庭在船上，我会尽力拯救他们，而不是先救自己。但是令人惊讶的是，5-8名成员家庭的存活率为0%。原因可能是他们在pclass=3的船舱？##


'''老人和孩子的数量对结果影响'''
#老人和孩子，住哪个船舱
print(pd.crosstab(data.Parch,data.Pclass))
#还是三等仓较多

fig7,ax7=plt.subplots(figsize=(12, 7))
sns.barplot('Parch','Survived',data=data,ax=ax7)
ax7.set_title('Parch vs Survived')
plt.show()
##带着父母的乘客有更大的生存机会。然而，大于3人后随着数字的增加而减少。##
##在船上的家庭父母人数中有1-3个的人的生存机会是好的。独自一人也证明是致命的，当船上有4个父母时，生存的机会就会减少。##


'''船票价格的影响'''
print('Highest Fare was:',data['Fare'].max())
print('Lowest Fare was:',data['Fare'].min())
print('Average Fare was:',data['Fare'].mean())
#各等级船舱票价
fig8, ax8 = plt.subplots(1, 3, figsize=(20, 8))
sns.distplot(data[data['Pclass']==1].Fare, ax=ax8[0])
ax8[0].set_title('Fares in Pclass 1')
sns.distplot(data[data['Pclass']==2].Fare,ax=ax8[1])
ax8[1].set_title('Fares in Pclass 2')
sns.distplot(data[data['Pclass']==3].Fare,ax=ax8[2])
ax8[2].set_title('Fares in Pclass 3')
plt.show()
##价格可能和生存率相关，但因为是连续值所以需要后期特征工程处理##

'''
性别：与男性相比，女性的生存机会很高。

Pclass：有，第一类乘客给你更好的生存机会的一个明显趋势。对于pclass3成活率很低。对于女性来说，从pclass1生存的机会几乎是。

年龄：小于5-10岁的儿童存活率高。年龄在15到35岁之间的乘客死亡很多。

港口：上来的仓位也有区别，死亡率也很大！

家庭：有1-2的兄弟姐妹、配偶或父母上1-3显示而不是独自一人或有一个大家庭旅行，你有更大的概率存活。
'''

data.to_csv('InsuranceForcast_xgboost.data' + os.sep + 'preprocess1.csv')
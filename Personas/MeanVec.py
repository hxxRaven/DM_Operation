import gensim
import numpy as np

file_name = r'E:\DM_Operation\Personas\data\train_queryList_writefile-1w.csv'
model = gensim.models.word2vec.Word2Vec.load(r'E:\DM_Operation\Personas\data\word2vec_300-1w.model')

with open(file_name, 'r') as f:
    cur_index = 0
    lines = f.readlines()
    doc_vec = np.zeros((len(lines), 300)) #构造用户向量表
    for line in lines:
        word_vec = np.zeros((1, 300))
        words = line.strip().split(" ")
        wrod_num = 0
        for word in words:   #求模型平均向量
            if word in model:
                wrod_num += 1
                z = np.array([model[word]])
                word_vec += np.array([model[word]])
        doc_vec[cur_index] = word_vec / float(wrod_num)
        cur_index += 1
        # print(type(doc_vec))

#导入三个指标
genderlabel = np.loadtxt(open(r'E:\DM_Operation\Personas\data\train_gender.csv', 'r')).astype(int)
educationlabel = np.loadtxt(open(r'E:\DM_Operation\Personas\data\train_education.csv', 'r')).astype(int)
agelabel = np.loadtxt(open(r'E:\DM_Operation\Personas\data\train_age.csv', 'r')).astype(int)

def remove_zeros(x, y):
    nozero = np.nonzero(y)  #获取非零索引
    y = y[nozero]
    # x = np.array(x)
    x = x[nozero]
    print(type(x))
    return x, y

gender_train, genderlabel = remove_zeros(doc_vec, genderlabel)
age_train, agelabel = remove_zeros(doc_vec, agelabel)
education_train, educationlabel = remove_zeros(doc_vec, educationlabel)


#画图脚本
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


'''建立预测模型'''
#逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(gender_train, genderlabel, test_size=0.2, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(lr.score(X_test, y_test))
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Recall metric in the testing dataset: ", cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1]))
print("accuracy metric in the testing dataset: ", (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]))
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Gender-Confusion matrix')
plt.show()


#随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(gender_train, genderlabel, test_size=0.2, random_state=0)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5, max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print (rf.score(X_test,y_test))
cnf_matrix = confusion_matrix(y_test,y_pred)
print("Recall metric in the testing dataset: ", cnf_matrix[0,0]/(cnf_matrix[0,1]+cnf_matrix[0,0]))
print("accuracy metric in the testing dataset: ", (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]))
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Gender-Confusion matrix')
plt.show()

#堆叠模型
#第一阶段几个分类器得结果，结果作为第二阶段输入，继续用分类器。但因为数据已经在第一阶段分类中用过，再在第二阶段用可能过拟合
#所以第一阶段采取交叉验证，用每个验证集的结果组合成新的训练集供第二阶段的分离器使用

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

#三个分类器模型
clf1 = RandomForestClassifier(n_estimators=100,min_samples_split=5,max_depth=10)
clf2 = SVC()
clf3 = LogisticRegression()
basemodes = [
            ['rf', clf1],
            ['svm', clf2],
            ['lr', clf3]
            ]
#交叉构造
from sklearn.model_selection import KFold

models = basemodes

kf = KFold(n_splits=5, random_state=0)
S_train = np.zeros((X_train.shape[0], len(models)))
S_test = np.zeros((X_test.shape[0], len(models)))

for i, m in enumerate(models):
    clf = m[1]

    for train_idx, test_idx in kf.split(X_train):
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_val = X_train[test_idx]
        clf.fit(X_train_cv, y_train_cv)
        y_val = clf.predict(X_val)

        S_train[test_idx, i] = y_val
    S_test[:,i] = clf.predict(X_test)   #一轮下来，训练好的分类器测试X_test数据

final_clf = RandomForestClassifier(n_estimators=100)
final_clf.fit(S_train,y_train)
print (final_clf.score(S_test,y_test))

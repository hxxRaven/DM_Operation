'''
使用Gensim库建立word2vec词向量模型

参数定义：
sentences：可以是一个list
sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
window：表示当前词与预测词在一个句子中的最大距离是多少
alpha: 是学习速率
seed：用于随机数发生器。与初始化词向量有关。
min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
workers参数控制训练的并行数。
hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
iter： 迭代次数，默认为5
'''
#
from gensim.models import word2vec
import time

train_path = r'E:\DM_Operation\Personas\data\train_queryList_writefile-1w.csv'

with open(train_path, 'r') as f:
    user_list = []
    lines = f.readlines()
    for line in lines:
        cur_list = []
        line = line.strip() #字符串切分为可被split切成list的形式（去除头尾空格）
        data = line.split(" ")
        for i in data:
            cur_list.append(i)
        user_list.append(cur_list)

    start_time = time.perf_counter()
    model = word2vec.Word2Vec(user_list, size=300, window=10, workers=4)    #语料库二维数组可满足训练条件
    end_time = time.process_time()
    save_path = r'E:\DM_Operation\Personas\data\word2vec_300-1w.model'
    model.save(save_path)   #npy也可以，因为本质是个npy
    print('Training Time:',end_time-start_time)

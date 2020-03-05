import pandas as pd
import os, sys
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
'''
首先检查JData_User中的用户和JData_Action中的用户是否一致¶
保证行为数据中的所产生的行为均由用户数据中的用户产生（但是可能存在用户在行为数据中无行为）
'''
def user_action_check():
    #merge()函数，如果只合入一列，那么会那这一列和原dataframe的同列判断，最后保留一样的
    #如果合入两列，也只会保留一样的，但总体样本数量会以原dataframe为准，并新增一列数据
    df_user = pd.read_csv('data'+os.sep+'JData_User.csv', encoding='gbk')
    user_id = df_user.loc[ : , 'user_id'].to_frame()
    df_action_02 = pd.read_csv('data'+os.sep+'JData_Action_201602.csv', encoding='gbk')
    df_action_03 = pd.read_csv('data'+os.sep+'JData_Action_201603.csv', encoding='gbk')
    df_action_04 = pd.read_csv('data'+os.sep+'JData_Action_201604.csv', encoding='gbk')
    print('Is action of Feb. from User file? ', len(df_action_02) == len(df_action_02.merge(user_id)))
    print('Is action of Mar. from User file? ', len(df_action_02) == len(df_action_02.merge(user_id)))
    print('Is action of Apr. from User file? ', len(df_action_02) == len(df_action_02.merge(user_id)))

# user_action_check()
'''
Is action of Feb. from User file?  True
Is action of Mar. from User file?  True
Is action of Apr. from User file?  True
结论： User数据集中的用户和交互行为数据集中的用户完全一致
根据merge前后的数据量比对，能保证Action中的用户ID是User中的ID的子集
'''


'''
检查是否有重复记录¶
除去各个数据文件中完全重复的记录,可能解释是重复数据是有意义的，比如用户同时购买多件商品，同时添加多个数量的商品到购物车等
但在本数据中，用户行为被频繁记录，可能会导致重复数据量大
'''
def check_duplicate(filepath, filename):
    df_file = pd.read_csv(filepath, encoding='gbk')
    before = len(df_file)
    df_file.drop_duplicates(inplace=True)
    after = len(df_file)
    n = before - after
    print('Duplicates records of ', filename , ' are ',n)
    if n != 0:
        df_file.to_csv('processeddata'+os.sep+'processed_'+filename+'.csv', index=None)
    else:
        print('No dumplicates')

# check_duplicate('data'+os.sep+'JData_Action_201602.csv', 'Feb. action')
# check_duplicate('data'+os.sep+'JData_Action_201603.csv', 'Mar. action')
# check_duplicate('data'+os.sep+'JData_Action_201604.csv', 'Apr. action')
# check_duplicate('data'+os.sep+'JData_Comment.csv', 'Comment')
# check_duplicate('data'+os.sep+'JData_Product.csv', 'Product')
# check_duplicate('data'+os.sep+'JData_User.csv', 'User')
'''
Duplicates records of  Feb. action  are  2756093
Duplicates records of  Mar. action  are  7085038
Duplicates records of  Apr. action  are  3672710

Duplicates records of  Comment  are  0
No dumplicates
Duplicates records of  Product  are  0
No dumplicates
Duplicates records of  User  are  0
No dumplicates

结论：action的重复特别高，但其他数据很少有重复
'''


'''
看下重复的行为数据是由什么类型的行为产生的
'''
def type_of_dup(filepath):
    df = pd.read_csv('data'+os.sep+filepath, encoding='gbk')
    IsDuplicates = df.duplicated()
    df_dup = df[IsDuplicates]
    print(filepath+'duplicate infomation:')
    print(df_dup.groupby('type').count())

# type_of_dup('JData_Action_201602.csv')
'''
JData_Action_201602.csvduplicate infomation:
      user_id   sku_id     time  model_id     cate    brand
type                                                       
1     2176378  2176378  2176378         0  2176378  2176378
2         636      636      636         0      636      636
3        1464     1464     1464         0     1464     1464
4          37       37       37         0       37       37
5        1981     1981     1981         0     1981     1981
6      575597   575597   575597    545054   575597   575597
1类行为，5类行为的重复率也太大了
'''


'''
注册时间是京东系统错误造成，如果行为数据中没有在4月15号之后的数据的话，那么说明这些用户还是正常用户，并不需要删除。
检查是否存在注册时间在2016年-4月-15号之后的用户
'''
def user_after_415(filepath):
    df = pd.read_csv('data'+os.sep+filepath, encoding='gbk')
    df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'])
    error_len = len(df.loc[df['user_reg_tm'] >= '2016-4-15'])
    print('There are ',error_len)

def action_after_415(filepath):
    df = pd.read_csv('data'+os.sep+filepath, encoding='gbk')
    df['time'] = pd.to_datetime(df['time'])
    df_after = df.loc[df['time'] >= '2016-4-15']
    print(df_after)
    
# action_after_415('JData_Action_201602.csv')
'''
Empty DataFrame
Columns: [user_id, sku_id, time, model_id, type, cate, brand]
Index: []
行为并没产生错误日期数据，不用处理
'''


'''
行为数据中的user_id为浮点型，进行INT类型转换
看着都统一，而且节省空间
'''
def convert_int_userid(filepath):
    df_month = pd.read_csv('data'+os.sep+filepath,encoding='gbk')
    df_month['user_id'] = df_month['user_id'].apply(lambda x : int(x))
    print (df_month['user_id'].dtype)
    df_month.to_csv('processeddata'+os.sep+'inted_'+filepath+'.csv',index=None)
    print('finished')

# convert_int_userid('JData_Action_201602.csv')
# convert_int_userid('JData_Action_201603.csv')
# convert_int_userid('JData_Action_201604.csv')


def distinction(x):
    if x == u'15岁以下':
        x='1'
    elif x==u'16-25岁':
        x='2'
    elif x==u'26-35岁':
        x='3'
    elif x==u'36-45岁':
        x='4'
    elif x==u'46-55岁':
        x='5'
    elif x==u'56岁以上':
        x='6'
    return x

def cut_age():
    df_user = pd.read_csv('data'+os.sep+'JData_User.csv', encoding='gbk')
    df_user['age'] = df_user['age'].apply(distinction)
    print (df_user.groupby(df_user['age']).count())
    df_user.to_csv('processeddata'+os.sep+'JData_User_Aged.csv', index=None)

# cut_age()
''' 
user_id    sex  user_lv_cd  user_reg_tm
age                                         
-1     14412  14412       14412        14412
1          7      7           7            7
2       8797   8797        8797         8797
3      46570  46570       46570        46570
4      30336  30336       30336        30336
5       3325   3325        3325         3325
6       1871   1871        1871         1871
'''


'''
构建user_table
user_table特征包括:
user_id(用户id),age(年龄),sex(性别),
user_lv_cd(用户级别),browse_num(浏览数),
addcart_num(加购数),delcart_num(删购数),
buy_num(购买数),favor_num(收藏数),
click_num(点击数),buy_addcart_ratio(购买加购转化率),
buy_browse_ratio(购买浏览转化率),
buy_click_ratio(购买点击转化率),
buy_favor_ratio(购买收藏转化率)
'''

ACTION_201602_FILE = "data"+os.sep+"JData_Action_201602.csv"
ACTION_201603_FILE = "data"+os.sep+"JData_Action_201603.csv"
ACTION_201604_FILE = "data"+os.sep+"JData_Action_201604.csv"
COMMENT_FILE = "data"+os.sep+"JData_Comment.csv"
PRODUCT_FILE = "data"+os.sep+"JData_Product.csv"
USER_FILE = "data"+os.sep+"JData_User.csv"
USER_TABLE_FILE = "data"+os.sep+"User_table.csv"
ITEM_TABLE_FILE = "data"+os.sep+"Item_table.csv"

import numpy as np
from collections import Counter


#把user表里的数据取出来，准备合成大表
def get_from_user():
    df = pd.read_csv(USER_FILE, encoding='gbk', header=0)
    print(df.head())
    df = df[['user_id', 'age', 'sex', 'user_lv_cd']]
    return df


def merge_action_df():
    df_pre = []
    df_pre.append(get_from_data(name=ACTION_201602_FILE))
    df_pre.append(get_from_data(name=ACTION_201603_FILE))
    df_pre.append(get_from_data(name=ACTION_201604_FILE))
    df = pd.concat(df_pre, ignore_index=True)
    # df =get_from_data(name=ACTION_201602_FILE)
    #根据id，对各指标求和。concat三个月的表，很多用户会在各个月消费，所以每张表里都会有相同user_id活动，需要再次groupby（）,在求和前已经统计了各表次数，所这里直接求和就行）
    df = df.groupby(['user_id'], as_index=True).sum()
    #　构造转化率字段
    df['buy_addcart_ratio'] = df['buy_num'] / df['addcart_num']
    df['buy_browse_ratio'] = df['buy_num'] / df['browse_num']
    df['buy_click_ratio'] = df['buy_num'] / df['click_num']
    df['buy_favor_ratio'] = df['buy_num'] / df['favor_num']
    # 如果转化率大于１的话，转化率字段置为１(100%)
    df.loc[df['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df.loc[df['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df.loc[df['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df.loc[df['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.
    return df


#对action数据进行统计。使用chunk_避免内存溢出
def get_from_data(name, chunk_size=5000):
    reader = pd.read_csv(name, header=0, iterator=True, encoding='gbk')
    chunks = []
    loop =True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[['user_id', 'type']]
            chunks.append(chunk)    #chunks里存的是dataframe
        except StopIteration:
            loop =False
            print('Iteration is stopped')
            #读完或者错误的时候停止迭代，多次读，再凑成dataframe
    df = pd.concat(chunks, ignore_index=True)
    df = df.groupby(['user_id'], as_index=False).apply(add_user_counter_type)
    df= df.drop_duplicates('user_id')
    return df

def add_user_counter_type(group):
    b_type = group['type'].astype(int)  #转成int，方便Counter
    type_count = Counter(b_type)    #统计模块，统计各值出现次数
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_count[1]
    group['addcart_num'] = type_count[2]
    group['delcart_num'] = type_count[3]
    group['buy_num'] = type_count[4]
    group['favor_num'] = type_count[5]
    group['click_num'] = type_count[6]

    return group[['user_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

# user_base = get_from_user()
# user_behavior = merge_action_df()
# user_behavior = pd.merge(user_base, user_behavior, on=['user_id'], how='left')
# print(user_behavior.head())
# user_behavior.to_csv('processeddata'+os.sep+'UserInfo.csv', index=False)


'''
构建item_table特征包括:
sku_id(商品id),attr1,attr2,
attr3,cate,brand,browse_num,
addcart_num,delcart_num,
buy_num,favor_num,click_num,
buy_addcart_ratio,buy_browse_ratio,
buy_click_ratio,buy_favor_ratio,
comment_num(评论数),
has_bad_comment(是否有差评),
bad_comment_rate(差评率)
'''

#获取Product产品信息
def get_from_product():
    product_df = pd.read_csv(PRODUCT_FILE, header=0, encoding='gbk')
    return product_df

#三张action表的整合，和比例特征构建
def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(name=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(name=ACTION_201603_FILE))
    df_ac.append(get_from_action_data(name=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.groupby(['sku_id'], as_index=False).sum()

    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    df_ac.loc[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.loc[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.loc[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.loc[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.
    return df_ac

#对action中的数据进行统计
def get_from_action_data(name, chunk_size=50000):
    reader = pd.read_csv(name, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[["sku_id", "type"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    df_ac = pd.concat(chunks, ignore_index=True)
    #groupby()的一个group包含的dataframe都会送入add_type
    df_ac = df_ac.groupby(['sku_id'], as_index=False).apply(add_type_stat)
    df_ac = df_ac.drop_duplicates('sku_id')
    return df_ac


#对一商品的分组统计
def add_type_stat(group):
    behavior_type = group['type'].astype(int)
    type_cnt = Counter(behavior_type)
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]
    return group[['sku_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

# 获取评论中的商品数据,如果存在某一个商品有两个日期的评论，我们取最晚的那一个
def get_from_comment():
    df_cmt = pd.read_csv(COMMENT_FILE, header=0)
    df_cmt['dt'] = pd.to_datetime(df_cmt['dt'])
    #groupby()['dt].transform会使每个sku_id的日期都变成最大那个
    #因为索引没有变，所以和原dataframe的同索引的日期对比，如果是最大最后日期的则标记此索引为True
    idx = df_cmt.groupby(['sku_id'])['dt'].transform(max) == df_cmt['dt']
    print(idx.value_counts())
    df_cmt = df_cmt[idx]
    return df_cmt[['sku_id', 'comment_num',
                   'has_bad_comment', 'bad_comment_rate']]

# item_base = get_from_product()
item_behavior = merge_action_data()
# item_comment = get_from_comment()
# item_behavior = pd.merge(item_base, item_behavior, on=['sku_id'], how='left')
# item_behavior = pd.merge(item_behavior, item_comment, on=['sku_id'], how='left')
# item_behavior.to_csv('processeddata'+os.sep+'ItemInfo.csv', index=False)



'''数据清洗，洗掉无效的样本'''
user_table = pd.read_csv('processeddata'+os.sep+'UserInfo.csv')
print(user_table.describe())
''' 
             user_id            age            sex     user_lv_cd  
count  105321.000000  105318.000000  105318.000000  105321.000000
用户总数105321，但是只有105318用户有个人资料，可以删除以上用户
'''

delete_list = user_table[(user_table['age'].isnull()) & (user_table['sex'].isnull())].index
user_table.drop(delete_list, axis=0, inplace=True)


'''删除无任何行为的用户'''
df_naction = user_table[(user_table['browse_num'].isnull()) & (user_table['addcart_num'].isnull()) & (user_table['delcart_num'].isnull()) & (user_table['buy_num'].isnull()) & (user_table['favor_num'].isnull()) & (user_table['click_num'].isnull())]
user_table.drop(df_naction.index,axis=0,inplace=True)

'''删除无购买记录的用户'''
user_table = user_table[user_table['buy_num']!=0]

print(user_table.describe())


'''
爬虫用户，浏览购买比相差悬殊的
       buy_addcart_ratio  buy_browse_ratio  buy_click_ratio  buy_favor_ratio  
count       29483.000000      29483.000000     29483.000000     29483.000000  
mean            0.359802          0.018411         0.030435         0.861532  
std             0.319737          0.038200         0.136395         0.286888  
min             0.004184          0.000161         0.000067         0.010417  
25%             0.117647          0.003759         0.002358         1.000000  
50%             0.250000          0.007899         0.005025         1.000000  
浏览购买比和点击购买比的均值为0.018,0.030，最小值：0.0001,0.00006. 25%位置是0.003 0.002
因此这里认为浏览购买转换比和点击购买转换比小于0.0005的用户为惰性用户
'''
sp = user_table[user_table['buy_browse_ratio'] < 0.0005].index
print(len(sp))
user_table.drop(sp, axis=0, inplace=True)

bp = user_table[user_table['buy_click_ratio']<0.0005].index
print (len(bp))
user_table.drop(bp, axis=0, inplace=True)

user_table.to_csv('processeddata'+os.sep+'After_Cleaned_UserTable.csv', index=False)


'''
因为xgboost，可以将缺失值也当做一个评估树分值的标准，所以对于item的空值就暂不处理
如果跑逻辑回归等算法，还是需要填充缺失值的（虽然它自己会填充）
'''
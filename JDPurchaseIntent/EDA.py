import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ACTION_201602_FILE = "data"+os.sep+"JData_Action_201602.csv"
ACTION_201603_FILE = "data"+os.sep+"JData_Action_201603.csv"
ACTION_201604_FILE = "data"+os.sep+"JData_Action_201604.csv"
COMMENT_FILE = "data"+os.sep+"JData_Comment.csv"
PRODUCT_FILE = "data"+os.sep+"JData_Product.csv"
USER_FILE = "data"+os.sep+"JData_User.csv"
USER_TABLE_FILE = "data"+os.sep+"User_table.csv"
ITEM_TABLE_FILE = "data"+os.sep+"Item_table.csv"


'''周一到周日各天购买情况，因为周末人们的购物欲望可能很强烈'''

def get_from_action_time(name):
    df_ac = pd.read_csv(name, encoding='gbk')
    #type4 ,购买行为
    df_ac = df_ac[df_ac['type'] == 4]
    return df_ac[['user_id', 'sku_id', 'time']]

def merge_from_action():
    df_ac = []
    df_ac.append(get_from_action_time(name=ACTION_201602_FILE))
    df_ac.append(get_from_action_time(name=ACTION_201603_FILE))
    df_ac.append(get_from_action_time(name=ACTION_201604_FILE))
    df_ac = pd.concat(df_ac, ignore_index=True)
    return df_ac


#time先转时间格式，再取成周1~7
def time_convert(df_ac):
    df_ac['time'] = pd.to_datetime(df_ac['time'])
    df_ac['time'] = df_ac['time'].apply(lambda x : x.weekday() + 1)  #取成星期数的功能，但是从0开始到6

    return df_ac

df_ac = time_convert(merge_from_action())


#画图显示每星期几各方面购买情况
def plt_week_purchase(df_ac):
    # 周一到周日每天购买的用户个数
    df_user = df_ac.groupby('time')['user_id'].nunique()    #此时会group整体，但是使用apply（）则会被分为同time的多行样本
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['weekday', 'user_num']

    # 周一到周日每天购买商品种类数
    df_item = df_ac.groupby('time')['sku_id'].nunique() #Serices
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['weekday', 'item_num']

    #周一到周日每天的购买记录(本身存的就是记录数，所以直接size)
    df_log = df_ac.groupby('time', as_index=False).size()   #Series
    df_log = df_log.to_frame().reset_index()
    df_log.columns = ['weekday', 'user_item_num']

    bar_width = 0.2
    opacity = 0.4
    plt.bar(df_user['weekday'], df_user['user_num'], bar_width, alpha=opacity, color='c', label='user')
    plt.bar(df_item['weekday']+bar_width, df_item['item_num'],bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_log['weekday']+bar_width*2, df_log['user_item_num'],bar_width, alpha=opacity, color='m', label='user_item')
    plt.xlabel('weekday')
    plt.ylabel('number')
    plt.title('A Week Purchase Table')
    plt.xticks(df_user['weekday']+bar_width*3/2, (1,2,3,4,5,6,7))
    plt.legend(prop={'size':10})
    plt.show()

# plt_week_purchase(df_ac)
'''周末和周三的销售情况不好，与我的预期相反。时间考虑进去'''

'''--------------------------------------------------------------------------------------------------'''


'''
寻找周末购买量少的原因
一个月中各天购买数量
'''
#画图看下各天的情况
def plt_day_purchase(df_ac, title):
    #每天购买的用户个数
    df_user = df_ac.groupby('time')['user_id'].nunique()    #此时会group整体，但是使用apply（）则会被分为同time的多行样本
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['days', 'user_num']

    #每天购买商品种类数
    df_item = df_ac.groupby('time')['sku_id'].nunique() #Serices
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['days', 'item_num']

    #每天的购买记录(本身存的就是记录数，所以直接size)
    df_log = df_ac.groupby('time', as_index=False).size()   #Series
    df_log = df_log.to_frame().reset_index()
    df_log.columns = ['days', 'user_item_num']

    bar_width = 0.2
    opacity = 0.4
    plt.bar(df_user['days'], df_user['user_num'], bar_width, alpha=opacity, color='c', label='user')
    plt.bar(df_item['days']+bar_width, df_item['item_num'],bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_log['days']+bar_width*2, df_log['user_item_num'],bar_width, alpha=opacity, color='m', label='user_item')
    plt.xlabel('days')
    plt.ylabel('number')
    plt.title(title+' Everyday Purchase Table')
    plt.xticks(df_user['days']+bar_width*3/2, range(1, len(df_user['days']) + 1, 1))
    plt.legend(prop={'size':10})
    plt.show()

# #2月份情况
# df_02 = get_from_action_time(name=ACTION_201602_FILE)
# df_02['time'] = pd.to_datetime(df_02['time']).apply(lambda x : x.day)   #日期转化为天
# plt_day_purchase(df_02, 'feb')
#
# #3月份情况
# df_03 = get_from_action_time(name=ACTION_201603_FILE)
# df_03['time'] = pd.to_datetime(df_03['time']).apply(lambda x : x.day)   #日期转化为天
# plt_day_purchase(df_03, 'mar')
#
# #4月份情况
# df_04 = get_from_action_time(name=ACTION_201604_FILE)
# df_04['time'] = pd.to_datetime(df_04['time']).apply(lambda x : x.day)   #日期转化为天
# plt_day_purchase(df_04, 'apr')

'''
2月份有个明显下坡，3月份有个高峰，4月份后期有个销售高峰
'''

'''--------------------------------------------------------------------------------------------'''


'''
商品类别销售统计
周一到周日各商品类别销售情况
'''
def get_from_action_catogary(name):
    df_ac = pd.read_csv(name, encoding='gbk')
    #type4 ,购买行为
    df_ac = df_ac[df_ac['type'] == 4]
    return df_ac[["cate", "brand", "type", "time"]] #只要想要的结果

def merge_convert_time_catogary():
    df_ac = []
    df_ac.append(get_from_action_catogary(name=ACTION_201602_FILE))
    df_ac.append(get_from_action_catogary(name=ACTION_201603_FILE))
    df_ac.append(get_from_action_catogary(name=ACTION_201604_FILE))
    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac['time'] = pd.to_datetime(df_ac['time'])
    df_ac['time'] = df_ac['time'].apply(lambda x : x.weekday() + 1)
    return df_ac

def plot_category(df_ac):
    # 周一到周日每天购买商品类别数量统计
    df_product = df_ac['brand'].groupby([df_ac['time'],df_ac['cate']]).count()
    print(df_product)
    df_product=df_product.unstack() #把左边的第二层索引放回到上方
    print(df_product)
    df_product.plot(kind='bar',title='Cate Purchase Table in a Week',figsize=(14,10))
    plt.show()

# df_cat = merge_convert_time_catogary()
# print(df_cat.groupby('cate').count())
'''
brand   type   time
cate                     
4      9326   9326   9326
5      8138   8138   8138
6      6982   6982   6982
7      6214   6214   6214
8     13281  13281  13281
9      4104   4104   4104
10      189    189    189
11       18     18     18

分析：星期二买类别8的最多，星期天最少。
'''
# plot_category(df_cat)
'''
星期二买类别8的最多，星期天最少。
-----------------------------------------------------------------------------------------------------
'''


'''每月每天各类商品销售情况（只关注商品8）'''
def merge_convert_everytime_catogary():
    #2月
    df_ac2 = get_from_action_catogary(name=ACTION_201602_FILE)
    df_ac2['time'] = pd.to_datetime(df_ac2['time']).apply(lambda x: x.day)
    dc_cate2 = df_ac2[df_ac2['cate']==8]
    dc_cate2 = dc_cate2['brand'].groupby(dc_cate2['time']).count()
    dc_cate2 = dc_cate2.to_frame().reset_index()
    dc_cate2.columns = ['day', 'product_num']

    #3月
    df_ac3 = get_from_action_catogary(name=ACTION_201603_FILE)
    df_ac3['time'] = pd.to_datetime(df_ac3['time']).apply(lambda x: x.day)
    dc_cate3 = df_ac3[df_ac3['cate']==8]
    dc_cate3 = dc_cate3['brand'].groupby(dc_cate3['time']).count()
    dc_cate3 = dc_cate3.to_frame().reset_index()
    dc_cate3.columns = ['day', 'product_num']

    #4月
    df_ac4 = get_from_action_catogary(name=ACTION_201604_FILE)
    df_ac4['time'] = pd.to_datetime(df_ac4['time']).apply(lambda x: x.day)
    dc_cate4 = df_ac4[df_ac4['cate']==8]
    dc_cate4 = dc_cate4['brand'].groupby(dc_cate4['time']).count()
    dc_cate4 = dc_cate4.to_frame().reset_index()
    dc_cate4.columns = ['day', 'product_num']
    dc = []
    dc.append(dc_cate2)
    dc.append(dc_cate3)
    dc.append(dc_cate4)
    return dc

#绘图
def plot_cate8(dc_cate2, dc_cate3, dc_cate4):
    # 条形宽度
    bar_width = 0.2
    # 透明度
    opacity = 0.4
    # 天数
    day_range = range(1,len(dc_cate3['day']) + 1, 1)
    # 设置图片大小
    plt.figure(figsize=(14,10))

    plt.bar(dc_cate2['day'], dc_cate2['product_num'], bar_width,
            alpha=opacity, color='c', label='February')
    plt.bar(dc_cate3['day']+bar_width, dc_cate3['product_num'],
            bar_width, alpha=opacity, color='g', label='March')
    plt.bar(dc_cate4['day']+bar_width*2, dc_cate4['product_num'],
            bar_width, alpha=opacity, color='m', label='April')

    plt.xlabel('day')
    plt.ylabel('number')
    plt.title('Cate-8 Purchase Table')
    plt.xticks(dc_cate3['day'] + bar_width * 3 / 2., day_range)
    # plt.ylim(0, 80)
    plt.tight_layout()
    plt.legend(prop={'size':9})
    plt.show()

recived = merge_convert_everytime_catogary()
plot_cate8(recived[0], recived[1], recived[2])
'''
2月份对类别8商品的购买普遍偏低，3，4月份普遍偏高，3月15日购买极其多！
可以对比3月份的销售记录，发现类别8将近占了3月15日总销售的一半！
同时发现，3,4月份类别8销售记录在前半个月特别相似，除了4月8号，9号和3月15号。

8商品的值有很明显区别，在15号。这种变化悬殊的数据是否保留要看后面的实验
-------------------------------------------------------------------------------
'''


'''
添加一个查询特定用户对特定商品整个轨迹的功能
看下用户观察多久，多次什么行为后会买商品
'''
def spec_ui_action_data(name, user_id, item_id):
    df_ac = pd.read_csv(name, encoding='gbk')
    df_ac = df_ac[(df_ac['user_id'] == user_id) & (df_ac['sku_id'] == item_id)]
    #这里可以筛选，只选那些有购买的用户，在对比下没有购买的行为轨迹没看下去别对比
    return df_ac[['user_id', 'sku_id', 'time', 'type','cate', 'brand']]

def everymonth_user_comsume_explore(user, item):
    user_id = user
    item_id = item
    df_ac = []
    df_ac.append(spec_ui_action_data(ACTION_201602_FILE, user_id, item_id))
    df_ac.append(spec_ui_action_data(ACTION_201603_FILE, user_id, item_id))
    df_ac.append(spec_ui_action_data(ACTION_201604_FILE, user_id, item_id))
    df_ac = pd.concat(df_ac, ignore_index=False)
    print(df_ac.sort_values(by='time'))

everymonth_user_comsume_explore(266079, 138778)
'''
    user_id  sku_id                 time  type  cate  brand
0    266079  138778  2016-01-31 23:59:02     1     8    403
1    266079  138778  2016-01-31 23:59:03     6     8    403
15   266079  138778  2016-01-31 23:59:40     6     8    403
'''

'''目的是看5天后用户购买可能'''
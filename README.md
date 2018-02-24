# Rossmann-Store-Sales
kaggle_competition
暂时笔记： 

Rossmann sales predict:

Description:
predict up to six weeks in advance 
create effective staff schedules that increase productivity and motivation. 

Evaluation: RMSPE

1st Documentation: 
for each train and test record, the model should have features on recent data, temporal information and current trends. 
1. Featurs Selection/Extraction



jupyter notebook的部分magic命令：
%timeit func()
%%timeit
测算整个单元格的运行时间，用 %%timeit

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
直接在 notebook中呈现图形，应将内联后端与命令 %matplotlib inline 一起使用
在 %matplotlib inline 之后使用 %config InlineBackend.figure_format = ‘retina’ 来呈现分辨率较高的图像

%pdb 
命令 %pdb 开启交互式调试器。出错时，你能检查当前命名空间中的变量

__future__说明：
Python提供了__future__模块，把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性
比如py3里面的功能导入到py2里面。

Seaborn使用：


额外需要学习的信息：来自于唐宇迪的机器学习
#136 pandas
TYD_ML:
pd.date_range('2016-5-5',periods=10,freq='T')
pd.Series(np.random.randn(10), index=pd.data_range(pd.datetime(2016,1,1),periods=10))
time.truncate(before='2016-1-5')
time.truncate(after='2016-1-5')
pd.datetime(2016,7,10)
pd.Timestamp('2016-07-10')
pd.Timestamp('2016-07-10 10:15')
pd.Period('2016-1')
pd.Timedelta('1D')
pd.Period('2016-1-1 10:10')+pd.Timedelta('1H')
pd.Timestamp('2016-5-6 10:15') + pd.Timedelta('1D')

ts = pd.Series(range(10), pd.date_range('07-10-16 8:00', periods = 10, freq = 'H'))
ts.to_period()
ts['2016-07-10 08:30':'2016-07-10 11:45'] 
ts_period['2016-07-10 08:30':'2016-07-10 11:45'] 


#137 resample
# 数据重采样  
降采样:reshape  
升采样:reshape以后，会出现NaN这个时候，需要bfill/ffill/interpolate

ts.resample('M').sum()
ts.resample('3D').sum().resample('D').asfreq().bfill(limit=2,axis=0)
ts.resample('3D').sum().resample('D').asfreq().interpolate(method='linear')

#138 滑动窗口
某年的365天，每一天由对应的10天（举例为10天）的平均来代表。
%matplotlib inline
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

df.rolling(window=10).mean()
pd.date_range('7/1/2016',freq='D',periods=600)
plt.figure(figsize=(20,10))
df.plot(stype='r--')
df.rolling(window=10).mean().plot(style='b')

#139差分

平稳性：要求样本时间序列所得到的的拟合曲线在未来的一段时间内仍能顺着现有的形态‘惯性’地延续下去
平稳性要求序列的均值和方差不发生明显的变化

严平稳与弱平稳： 严平稳：期望与方差不随时间的变化而变化
弱平稳：期望和相关系数不变
X_t要依赖过去的信息，所以需要依赖性

差分法：时间序列在t与t-1时刻的差值   
一阶差分与二阶差分，二阶是在一阶的基础上做的

#140 ARIMA
1.
自回归模型（AR）
	描述当前值与历史值之间的关系，用变量自身的历史事件数据对自身进行预测
	自回归模型必须满足平稳性的要求
	p阶自回归过程的公式定义： y_t = u + sum(gamma_i * y_i-1) + epsilon_t
	i 为5，就是和前5天有关，p阶就是和前p天有关心
	y_t是当前值，u是常数项，P是阶数， gamma_i是自相关系数，epsilon是误差

自回归模型的限制：
	自回归模型是用自身的数据进行预测
	必须具有平稳性
	必须具有自相关性，如果自相关系数(phi_i)小于0.5，则不宜采用
	自回归只适用于预测与自身前期相关的现象

移动平均模型（MA）
	关注的是自回归模型中的误差项的累加
	q阶自回归过程的公式定义： y_t = μ + epsilon_t + sum( theta_i * epsilon_t-1))
	移动平均法能有效地消除预测中的随机波动

ARIMA 自回归移动平均模型
	自回归与移动平均的结合
	公式定义：y_t = μ + sum()



#141 相关函数评估方法
ACF:自相关函数(autocorrelation function)
	有序的随机变量序列与其自身比较
	自相关函数反映了同一序列在不同时序的取值之间的相关性
	公式ACF(k) = ρ_k = Cov(y_t, y_t-k)/(Var(y_t))
	P_k 的取值范围[-1,1]
	k：阶数

PACF：偏相关函数(partial autocorreltaion function) 做的更绝对，不会掺杂其他
	对于一个平稳AR(ρ)模型，求出滞后k自相关系数p(k)时实际上得到并不是x(t)与x(t-k)之间单纯的相关系数
	ACF包含了其他变量的影响
	而PACF是严格在这个两个变量之间的相关性

#142
首先来导入相关的库，并设置好参数：
#自动重新加载更改的模块
%load_ext autoreload 
%autoreload 2
#matplotlib设置matplotlib的工作方式
%matplotlib inline
# 来呈现分辨率较高的图像
%config InlineBackend.figure_format='retina'

from __future__ import absolute_import, division, print_function
import sys
import os
import pandas as pd
import numpy as np

# TSA from Statsmodels
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
import statsmodels.tsa.api as smt 
# Display and Plotting
import matplotlib.pylab as plt 
import seaborn as sns 

pd.set_option('display.float_format', lambda x:'%.5f' %x)
np.set_printoptions(precision=5, suppress=True)

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
# seaborn plotting style
sns.set(style='ticks', context='poster')

设置完毕。
开始处理数据

Sentiment = pd.read_csv('data/sentiment.csv',index_col=0, parse_date=[0])
Sentiment.head()
sentiment_short = Sentiment.loc['2005':'2016']
sentiment_short.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title('Consumer Sentiment')
#通过往 despine() 中添加参数去控制边框
#sns.despine(left=True) # 删除左边边框
sns.despine()

#一阶差分二阶差分
sentiment_short['diff_1'] = sentiment_short['UMCSENT'].diff(1)
sentiment_short['diff_2'] = sentiment_short['diff_1'].diff(1)
sentiment_short.plot(subplots=True, figsize=(18,12))

#ARIMA:
确定 ARIMA(p, d, q)阶数确定：
	   ACF    PACF	
AR(p): 衰减趋于零  p阶后结尾
MA(q): q阶后结尾   衰减趋于零
ARM(p, q):q阶后衰减衰减趋于零   p阶衰减趋于零
ARIMA建模流程：
1. 将序列平稳（差分确定d）
2. p和q阶数确定: ACF与PACF
3. ARIMA(p,d,q)
阶数 差分 阶数 
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sentiment_short, lags = 20, ax=ax1)
ax1.xaxis.set_tricks_position('bottom')


#144 股票预测案例
import pandas as pd 
import datetime
import matplotlib.pylab as plt 
import seaborn as sns 
from matplotlib.pylab import style 
from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.graphics.tsaplots import plot_act, 

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
stockFile = './data/T10yr.csv'
stock = pd.read_csv(stockFile, index_col = 0, parase_dates = [0])
stock.head(5)
stock.tail(5)

stock_week = stock['Close'].resample('W-MON').mean()
stock_train = stock_week.loc['2000':'2015']
stock_train.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title('Stock Close')
sns.despine()

stock_diff = stock_trian.diff()
stock_diff = stock.diff.dropna()

model.ARIMA(stock_trian, order=(1,1,1), freq='W_MON')






class DataFrameImputer(TransformerMixin):
	def __init__(self):

	def fit(self, X, y=None):
		self.fill = pd.Series([X[c].value_counts().index[0]])





对于离散型特征，基于树的方法是不需要使用one-hot编码的，例如随机森林等。
基于距离的模型，都是要使用one-hot编码，例如神经网络.

s = pd.Series(['1.0', '2', -3])
pd.to_numeric(s,downcast='float')
pd.to_numeric(s,downcast='signed')
pd.to_numeric(s,downcast='integer')

import pandas as pd
import numpy as np
import csv
import datetime
# import xgboost as xgb
import itertools
import operator
import warnings 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation 
import matplotlib.pyplot as plt
plot=True
goal = 'Sales'
myid = 'Id'


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y!=0
    w[ind]=1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w*(yhat-y)**2))
    return rmspe

def rmspe_xg(yhat, y):
    y=y.get_label()
    y = np.exp(y)-1
    yhat = np.exp(y)-1
    w = ToWeight()
    rmspe = np.sqrt(np.mean(w*(y-yhat)**2))
    return 'rmspe', rmspe

train_org = pd.read_csv('./train.csv')
test_org = pd.read_csv('./test.csv')
store = pd.read_csv('./store.csv')
train_org.head()
test_org.head()
store.head()

train=pd.merge(train_org, store, how='left',on='Store')

test=pd.merge(test_org, store, how='left',on='Store')

features = test.columns.tolist()

features_numeric = test.columns[test.dtypes!='object'].tolist()
features_non_numeric=test.columns[test.dtypes=='object'].tolist()

features_numeric
features_non_numeric

train = train[train['Sales']>0]
train.shape
train.head()

train.PromoInterval.value_counts()

for data in [train, test]:
    data['year'] = train['Date'].apply(lambda x :float(x.split('-')[0]))
    data['month'] = train['Date'].apply(lambda x :float(x.split('-')[1]))
    data['day']  = train['Date'].apply(lambda x :float(x.split('-')[2]))
    data['promo_type1'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
    data['promo_type2'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
    data['promo_type3'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
    data['promo_type3'] = data.PromoInterval.apply(lambda x: 1 if isinstance(x, float) else 0)

noisy_features=[myid,'Date','PromoInterval']
features = [x for x in features if x not in noisy_features]
features_non_numeric = [x for x in features_non_numeric if x not in noisy_features]


len(train['Store'].value_counts())#店的个数
len(train['Date'].value_counts())#天的个数


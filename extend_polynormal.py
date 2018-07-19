# -*- coding:utf-8 -*-
# @Author  : cs
# @Time    : 2018/7/19 11:00
# @File    : extend_polynormal
# @describe: 
#1、特征工程之多项式扩展
#2、过拟合概念解释
#3、sklearn.Pipeline的用法
# 多项式扩展（多项式曲线拟合）：将特征与特征之间进行融合，从而形成新的特征的一个过程；
# 从数学空间上来讲，就是将低维度空间的点映射到高维度空间中。
# 更容易找到隐含的特性  属于特征工程的某一种操作  实际中用得比较少，一般用高斯扩展比较多 后面会讲
# 作用：通过多项式扩展后，我们可以提高模型的准确率或效果
# 过拟合：模型在训练集上效果非常好，但是在测试集中效果不好
# 多项式扩展的时候，如果指定的阶数比较大，那么有可能导致过拟合
# 从线性回归模型中，我们可以认为训练出来的模型参数值越大，就表示越存在过拟合的情况

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path,sep=';',low_memory=False)

new_df = df.replace('?',np.nan)
datas = new_df.dropna(axis=0,how='any')

# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
X = datas.iloc[:,0:2]
Y = datas['Voltage']

#非数值型数据转换成数值型数据

def date_formate(dt):
    t_struct = time.strptime(' '.join(dt),'%d/%m/%Y %H:%M:%S')
    return (t_struct.tm_year,t_struct.tm_mon,t_struct.tm_mday,t_struct.tm_hour,t_struct.tm_min,t_struct.tm_sec)

X = X.apply(lambda x: pd.Series(date_formate(x)),axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

## 时间和电压之间的关系(Linear-多项式)
# Pipeline：管道的意思，讲多个操作合并成为一个操作
# Pipleline总可以给定多个不同的操作，给定每个不同操作的名称即可，执行的时候，按照从前到后的顺序执行
# Pipleline对象在执行的过程中，当调用某个方法的时候，会调用对应过程的对应对象的对应方法
# eg：在下面这个案例中，调用了fit方法，
# 那么对数据调用第一步操作：PolynomialFeatures的fit_transform方法对数据进行转换并构建模型
# 然后对转换之后的数据调用第二步操作: LinearRegression的fit方法构建模型
# eg: 在下面这个案例中，调用了predict方法，
# 那么对数据调用第一步操作：PolynomialFeatures的transform方法对数据进行转换
# 然后对转换之后的数据调用第二步操作: LinearRegression的predict方法进行预测
models=[
    Pipeline(steps = [('Poly',PolynomialFeatures()),# 给定进行多项式扩展操作， 第一个操作：多项式扩展
              ('Linear',LinearRegression(fit_intercept=False))
              ])
]
model = models[0]
# print(type(model))

#模型训练
t = np.arange(len(X_test))
N = 5
d_pool = np.arange(1,N,1)#阶数
m = d_pool.size

clrs = []

for c in np.linspace(12345678,255,m):
    clrs.append('#%06x' % int(c))
# print(clrs)

line_width=3

plt.figure(figsize=(12,6),facecolor='w')
for i,d in enumerate(d_pool):
    plt.subplot(N-1,1,d)
    plt.plot(t,Y_test,'k-',label=u'真实值',ms=20,zorder=N)
    ### 设置管道对象中的参数值，Poly是在管道对象中定义的操作名称， 后面跟参数名称；中间是两个下划线
    model.set_params(Poly__degree=d)
    # pl = model.get_params()['Poly']
    # pl.set_params(degree = d)
    # model.set_params(Linear__fit_intercept=True)
    model.fit(X_train,Y_train)
    # Linear是管道中定义的操作名称
    # 获取线性回归算法模型对象
    lr = model.get_params()['Linear']
    output = u'%d阶,系数为：'%d
    # 判断lr对象中是否有对应的属性
    if hasattr(lr,'alpha_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'alpha=%.6f, ' % lr.alpha_) + output[idx:]
    if hasattr(lr, 'l1_ratio_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'l1_ratio=%.6f, ' % lr.l1_ratio_) + output[idx:]
    print(output, lr.coef_.ravel())

    #模型结果预测
    y_hat = model.predict(X_test)
    #计算评估值
    s = model.score(X_test,Y_test)

    #画图
    z = N - 1 if (d == 2) else 0
    label = u'%d阶，准确率：%0.3f'% (d,s)
    plt.plot(t,y_hat,clrs[i],lw=line_width,alpha=0.75,label=label,zorder=z)
    plt.legend(loc='upper left')
    plt.grid(b=True)
    plt.ylabel(u'%d阶结果'%d,fontsize=12)

plt.suptitle(u'线性回归预测时间和电压之间的多项式关系',fontsize=20)
plt.grid(b=True)
plt.show()


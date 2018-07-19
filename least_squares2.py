# -*- coding:utf-8 -*-
# @Author  : cs
# @Time    : 2018/7/19 9:01
# @File    : least_squares2
# @describe:
# 1、最小二乘法搭建模型，通过自建模型进行预测
# 2、模型持久化、加载使用
# 注意：如果进行特征数据标准化则需要截距这个参数，否则会出现偏移
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
import time

##设置字符集 防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#加载数据
path = './datas/household_power_consumption_1000.txt'
df = pd.read_csv(path,sep=';',low_memory=False)
# print(df.head(3))

##功率和电流之间的关系
X2 = df.iloc[:,2:4]
# print(X2.shape)
Y2 = df.iloc[:,5]

##数据分割
X_train,X_test,Y_train,Y_test = train_test_split(X2,Y2,test_size=0.2,random_state=0)

#模型对象创建
ss= StandardScaler()
#训练模型并转换训练集
X_train = ss.fit_transform(X_train)#得到数组
X_test = ss.transform(X_test)
# print(type(X_train))

#构建预测算法模型
##将数组转换成矩阵
X=np.mat(X_train)
Y=np.mat(Y_train).reshape(-1,1)#转换成n行1列

##计算θ
theta = (X.T*X).I*X.T*Y
print(theta)

##对测试数据进行预测 y=Xθ
y_hat = np.mat(X_test)*theta
# print(y_hat)

#画图
t=np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t,Y_test,'r-',linewidth=2,label=u'真实值')
plt.plot(t,y_hat,'b',linewidth=2,label=u'预测值')
plt.legend(loc='upper left')
plt.title(u'线性回归预测功率与电流之间的关系',fontsize=20)
plt.grid(b=True)
plt.show()


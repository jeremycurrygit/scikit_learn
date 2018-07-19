# -*- coding:utf-8 -*-
# @Author  : cs
# @Time    : 2018/7/19 9:58
# @File    : model_persistence
# @describe: 
# 模型持久化和加载使用

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

##设置字符集 防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#加载数据
path = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path,sep=';',low_memory=False)
#数据异常处理
new_df = df.replace('?',np.nan)
datas = new_df.dropna(axis=0,how='any')
#数据集划分
# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
X = datas.iloc[:,2:4]
Y = datas.iloc[:,5]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#创建训练模型
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train,Y_train)

#利用模型进行预测
y_predict = lr.predict(X_test)


#模型持久化/模型保存
# 在机器学习部署的时候，实际上其中一种方式就是将模型进行输出；另外一种方式就是直接将预测结果输出数据库
# 模型输出一般是将模型输出到磁盘文件
from sklearn.externals import joblib
# 保存模型要求给定的文件所在的文件夹必须存在
joblib.dump(ss,'result/data_ss.model')
joblib.dump(lr,'result/data_lr.model')

#加载模型
ss_load = joblib.load('result/data_ss.model')
lr_load = joblib.load('result/data_lr.model')

#使用加载的模型进行预测
data=[[10,11]]
data = ss_load.transform(data)
y_hat = lr_load.predict(data)
print(u'加载模型预测值：',y_hat)

#画图
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t,Y_test,'r-',linewidth=2,label=u'真实值')
plt.plot(t,y_predict,'b',linewidth=2,label=u'预测值')
plt.legend(loc='upper left')
plt.title(u'线性回归预测功率与电流之间的关系',fontsize=20)
plt.show()


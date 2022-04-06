from numpy import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
boston = load_boston()

x, _, y, _ = train_test_split(boston['data'], boston['target'], shuffle=True, test_size=1)

'''制造缺值'''
missing_rate = 0.5
n_miss_samples = int(x.shape[0]*x.shape[1]*missing_rate)
rng = np.random.RandomState(0) #固定种子，正常用np.random.randint就可以
missing_features = rng.randint(0, x.shape[1], n_miss_samples)
missing_samples = rng.randint(0, x.shape[0], n_miss_samples)
x[missing_samples, missing_features] = np.nan #对应索引位置变为nan
x = pd.DataFrame(x)
print(x)

def mean_fill(x):
    '''均值填充缺值'''
    #实例化，将np.nan以均值进行填充，如果strategy是constant，通过fill_value赋值
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0) 
    x_fill = imp_mean.fit_transform(x) #变换
    print(pd.DataFrame(x_fill).isna().sum()) #查看是否还有nan值
    rfr = RandomForestRegressor()
    score = cross_val_score(rfr, x_fill, y, cv=10)
    print(score)

def randomforest_fill(x):
    '''随机森林填充缺值（实质是预测缺值）'''
    sortidx = np.argsort(x.isnull().sum(axis=0)).values #按缺失值数量小到大对特征排序
    print(sortidx)
    x = x.to_numpy()
    total_idx = [i for i in range(x.shape[1])]
    print(x.shape)
    for idx in sortidx:
        temp = total_idx[:]
        temp.pop(idx)
        print(idx, temp)
        leave = x[:, temp]
        select = x[:, idx]
        print(leave.shape, select.shape)
        quit()
randomforest_fill(x)
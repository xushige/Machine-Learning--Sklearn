'''XGBoost'''

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from  sklearn.model_selection import cross_val_score, KFold, train_test_split, ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
XgbBoost与sklearn实操有所区别:

xgb.Dmatrix()              # 数据读取
param = {}                 # 参数设置
model = xgb.train(param)   # 训练
model.predict()            # 预测
'''

data = load_boston()
x, y = data['data'], data['target']
x = StandardScaler().fit_transform(x)
print('data.shaep: %s     label.shape: %s' % (x.shape, y.shape))

print('===============================Start train-test-split===============================')
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, shuffle=True)
xgb_model = xgb.XGBRegressor().fit(xtrain, ytrain)
pre = xgb_model.predict(xtest)
score = xgb_model.score(xtest, ytest) # R2_score
print('XGBoost\'s R2_score in Boston dataset : [%.4f]'%(score))

mse_error = mean_squared_error(ytest, pre)
print('XGBoost\'s MSE error in Boston dataset : [%.4f]'%(mse_error))

feature_importance = sorted(zip(xgb_model.feature_importances_, data['feature_names']), reverse=True)
print('\nTree model中可以调用feature importance进行特征选择，如下：\n\n%s\n'%feature_importance)

print('===============================Start Cross-val===============================')
cross_score = cross_val_score(xgb.XGBRegressor(), x, y, cv=ShuffleSplit(5, test_size=0.3)) # R2_score与model.score保持一致，回归模型：R2-score；分类模型：Accuracy
print('XGBoost\'s cross-val-R2-score in Boston dataset : [%.4f]'%(cross_score.mean()))
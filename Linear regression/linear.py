from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''数据预处理（加载，归一化，切分）'''
data = load_wine()
x, y = data['data'], data['target']
print('data shape: %s     label shape: %s' % (x.shape, y.shape))
feature_names = data['feature_names']
x = pd.DataFrame(x, columns=feature_names)
x = StandardScaler().fit_transform(x)
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, shuffle=True)

'''回归模型建立'''
# fit_intercept: 是否计算截距，默认false
# normalize：按列进行归一化，相当于StandardScaler，默认false
# copy_X：在X.copy()上进行计算，默认True
# n_jobs：用于计算的作业数，如果输入-1，表示使用全部cpu计算，默认None
linear = LinearRegression(fit_intercept=True).fit(trainx, trainy)
score = linear.score(testx, testy)
coef = sorted(list(zip(feature_names, linear.coef_)), key=lambda x:x[1], reverse=True)
intercept = linear.intercept_
print('====================================================================================================')
print("When split-style is train/test==7:3, the score of linearregression is: [%.4f]\nThe intercept is: [%.4f]\nImportance condition:\n%s"%(score, intercept, coef))
print('====================================================================================================')
cross_score = cross_val_score(LinearRegression(), x, y, cv=ShuffleSplit(5, test_size=0.3))
print("When split-style is 5-fold, the score of linearregression is: [%.4f]"%(cross_score.mean()))
print('====================================================================================================')

'''回归模型评价指标（是否预测正确【precision】，是否拟合足够多的信息【recall】）'''
# 是否预测正确：【precision】
# Mean Squared Error（MSE）: 均方误差, PS：【sklearn中MSE始终为负，表示损失loss】
print("MSE SCORE:")
mse_score = mean_squared_error(linear.predict(testx), testy)
print('When split-style is train/test==7:3, MSE score is: %.4f'%(mse_score))
cross_mse_score = cross_val_score(LinearRegression(), x, y, cv=ShuffleSplit(5, test_size=0.3), scoring='neg_mean_squared_error')
print("When split-style is 5-fold, the MSE core of linearregression is: [%.4f]"%(cross_mse_score.mean())) 
print('====================================================================================================')

# 是否拟合足够多信息【recall】
# 1. R2_score: 直接调用model.score默认为r2score
print("R2 SCORE:")
print('When split-style is train/test==7:3, R2 score is: %.4f'%(score))
cross_r2_score = cross_val_score(LinearRegression(), x, y, cv=ShuffleSplit(5, test_size=0.3), scoring='r2')
print("When split-style is 5-fold, the R2 core of linearregression is: [%.4f]"%(cross_r2_score.mean())) 
print('====================================================================================================')

# 可解释性方差
evs = explained_variance_score(testy, linear.predict(testx))
print('EVS of Linear Regression: %.4f'%(evs))

'''可视化'''
plt.plot(testy, label='True')
plt.plot(linear.predict(testx), label='Pred')
plt.legend()
plt.ylim(-4, 6)
plt.title('Linear Regression Effect')
plt.savefig('linear_regression_effect')
plt.clf()
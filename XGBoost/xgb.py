'''XGBoost'''

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from  sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split, learning_curve
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
cross_score = cross_val_score(xgb.XGBRegressor(), x, y, cv=KFold(5, shuffle=True)) # R2_score与model.score保持一致，回归模型：R2-score；分类模型：Accuracy
cross_score_rf = cross_val_score(RandomForestRegressor(), x, y, cv=KFold(5, shuffle=True)) # random forest
cross_score_linear = cross_val_score(LinearRegression(), x, y, cv=KFold(5, shuffle=True)) # linear regression
print('R2_score metrics in Boston dataset\n     XGBoost : [%.4f]\n     Random Forest: [%.4f]\n     Linear Regression : [%.4f]'%(cross_score.mean(), cross_score_rf.mean(), cross_score_linear.mean()))


print('===============================Start Visulization===============================')
def plot_learning_curve(estimator, title, x, y, cv=None):
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, shuffle=True, cv=cv)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='r', label='Training Score', linewidth=3)
    plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='black', label='Test Score', linewidth=3)
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.grid()
    plt.savefig(title)
    plt.clf()
plot_learning_curve(xgb.XGBRegressor(n_estimators=100), 'XGBoost_learning_curve', x, y, cv=KFold(5, shuffle=True))

def nestimator_selection_vis(nestimators=None, subsamples=None, etas=None, xlabel=None):
    if nestimators != None:
        params = nestimators
    elif subsamples != None:
        params = subsamples
    elif etas != None:
        params = etas
    scores = [] # 得分
    vars = [] # 方差
    ges = [] # 泛化误差
    for param in params:
        if isinstance(nestimators, list):
            model = xgb.XGBRegressor(n_estimators = param)
        elif isinstance(subsamples, list):
            model = xgb.XGBRegressor(subsample = param)
        elif isinstance(etas, list):
            model = xgb.XGBRegressor(eta = param)
        result = cross_val_score(model, x, y, cv=KFold(5, shuffle=True))
        scores.append(result.mean())
        vars.append(result.var())
        ges.append((1-result.mean())**2 + result.var())
    
    idx_maxscore = scores.index(max(scores))
    idx_minvar = vars.index(min(vars))
    idx_minge = ges.index(min(ges))

    print('按照 最大 R2_score 来挑选，此时%s： 【%f】  score: %.4f  var: %.4f  ge: %.4f'%(xlabel, params[idx_maxscore], scores[idx_maxscore], vars[idx_maxscore], ges[idx_maxscore]))
    print('按照 最小 Variance 来挑选，此时%s： 【%f】  score: %.4f  var: %.4f  ge: %.4f'%(xlabel, params[idx_minvar], scores[idx_minvar], vars[idx_minvar], ges[idx_minvar]))
    print('按照 最小 Generalization error 来挑选，此时%s： 【%f】  score: %.4f  var: %.4f  ge: %.4f\n'%(xlabel, params[idx_minge], scores[idx_minge], vars[idx_minge], ges[idx_minge]))
    plt.plot(params, scores, 'o-', linewidth=2, label='score', color='#CB181B')
    plt.fill_between(params, np.array(scores)-np.array(vars), np.array(scores)+np.array(vars), color='#CB181B', alpha=0.3)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('R2_score')
    plt.title(xlabel+" selcetion")
    plt.savefig(xlabel+' selcetion')
    plt.clf()
'''n_estimators参数: 树数量'''
nestimator_selection_vis([*range(30, 200, 10)], None, None, 'n_estimator')
'''subsample参数: 随机抽样过程中抽取样本比例, 范围 (0, 1]'''
nestimator_selection_vis(None, np.linspace(0.5, 1, 10).tolist(), None, 'subsample')
'''eta参数: XGB的学习率, 默认0.3'''
nestimator_selection_vis(None, None, np.linspace(0.2, 0.5, 20).tolist(), 'eta')

print('===============================Start Grid Search===============================')
'''网格搜索，grid search'''
params = {
    'n_estimators': [*range(30, 200, 10)],
    'subsample': np.linspace(0.5, 1, 10),
    'eta': np.linspace(0.2, 0.7, 20)
}

grid = GridSearchCV(xgb.XGBRegressor(), param_grid=params, cv=KFold(5, shuffle=True)).fit(x, y)
print('Best params: %s\nBest score: %.4f'%(grid.best_params_, grid.best_score_.mean()))
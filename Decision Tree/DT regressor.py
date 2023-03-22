from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score  #【k折交叉验证打分】


boston_data = load_boston()
print(boston_data.keys())
regressor = tree.DecisionTreeRegressor()

#estimator:实例化的模型
#X和y：总X数据和总标签，用于交叉验证
#cv：分为多少份进行交叉验证
#scoring：选择score机制，默认是R平方，可以选择MSE等等之类，函数最后返回cv个score的列表；
a = cross_val_score(estimator=regressor, X=boston_data.data, y=boston_data.target, cv=10)
print(a)

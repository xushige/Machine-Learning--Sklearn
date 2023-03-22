import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data = pd.read_csv('titanic_data/train.csv')
print(data.info(), data.head(10))

#删除无影响的特征
data.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
print(data.info())

#填补大量年龄缺失
data['Age'].fillna(data['Age'].mean(), inplace=True)
print(data.info())

#数据对齐，木桶原理，按最短的来，去除含nan的行/列
data.dropna(axis=0, inplace=True)
print(data.info())

#将object类型转换为int标签型, unique后是数组形式object型标签
transform = [data['Embarked'], data['Sex']]
for eachitem in transform:
    eachitem.replace(eachitem.unique(), [i for i in range(len(eachitem.unique()))], inplace=True)
print(data.info())

#取出y标签，得到x数据
y = data['Survived']
data.drop('Survived', axis=1, inplace=True)
print(data.head(10), '\n', y)

#交叉验证则不需要切分数据集
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.3)

#网格搜索：枚举试验出得分最高的参数组合，填入实例化的模型中
parameters = {
    'criterion': ('gini', 'entropy'),
    'max_depth': [*range(1, 5)],
    'min_samples_leaf': [*range(1, 10, 2)],
    'min_impurity_decrease': np.arange(0, 0.5, 0.1)
}
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=3, min_impurity_decrease=0)
Grid_Search = GridSearchCV(estimator=clf, param_grid=parameters, cv=10).fit(data, y)
print(Grid_Search.best_params_, Grid_Search.best_score_)


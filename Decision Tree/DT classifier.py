from sklearn import tree
from sklearn.datasets import load_wine #红酒数据
from sklearn.model_selection import train_test_split
import graphviz

#输入数据
wine = load_wine() #字典
print(wine.keys())
#按比例随机切割数据
X_train, X_test, Y_train, Y_test = train_test_split(wine.data, wine.target, test_size=0.3)

#分类器实例化
#criterion:选择entropy或者gini进行不纯度的计算，信息熵相比基尼系数计算速度慢，因为有对数，但是分枝更精细
#random_state:选择随机种子(int型），设置固定的随机种子则每次生成决策树的过程都一致，打分结果也是一样
#splitter:输入best或者random，选择random更不容易过拟合
#max_depth:限制树的最大深度，超过该深度的树枝全部减除，防止过拟合。
#min_samples_split:一个节点必须包含的最少样本数，不然就不会向下分支
#min_samples_leaf:每个即将分支出来的子节点必须包含的最少样本数，若少于该数量，父节点取消分支
#max_features:设置最大可用的特征，限制过拟合，和max_depth类似
#min_impurity_decrease:限制信息增益大小，信息增益小于该值的分枝不会发生（子节点的信息熵高于父节点）。
#class_weight:让少量的标签更多的权重，参数默认为None，自动给所有标签相同权重
classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=20, splitter='random', max_depth=3, min_samples_split=3, min_samples_leaf=3)
#训练
classifier = classifier.fit(X_train, Y_train)
#打分
#score默认的的打分机制是 R平方=【1-u/v】，其中u为【预测标签】与【分类标签】差值的平方之和，v为【分类标签】与【所有标签的平均值】差值的平方之和
score = classifier.score(X_test, Y_test)
train_score = classifier.score(X_train, Y_train)
print('test_acc:%f, train_acc:%f' % (score, train_score))

#每个测试样本所在决策树中的节点的索引，每个测试样本的预测标签值
apply = classifier.apply(X_test)
predict = classifier.predict(X_test)
print(apply, predict)

#特征重要性：判断用到了哪些特征
feature_importance = dict(zip(wine.feature_names, classifier.feature_importances_))
print(feature_importance)

#将分类器数据送入，进行决策树可视化
#filled是按类分颜色，每一类按不纯度分颜色深度；rounded是每一个结点变成圆角矩形
tree_data = tree.export_graphviz(classifier, feature_names=wine.feature_names, class_names=wine.target_names, filled=True, rounded=True)
graph = graphviz.Source(tree_data)



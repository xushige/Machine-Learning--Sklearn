from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

wine_data = load_wine()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine_data.data, wine_data.target, test_size=0.3)
# 实例化，还没有生成树,fit后生成决策树。
# 随机森林中random_state形成了25个固定random_state的树，并不是25个一样random_state的树
# bootstrap：n个样本
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=3, random_state=0)
rfc = RandomForestClassifier(n_estimators=35, criterion='entropy', max_depth=4, min_samples_leaf=1, random_state=0)


def normal_compare():
    global clf, rfc
    clf = clf.fit(Xtrain, Ytrain)
    rfc = rfc.fit(Xtrain, Ytrain)

    score_tree = clf.score(Xtest, Ytest)
    score_forest = rfc.score(Xtest, Ytest)
    print('Tree normal test socre: 【%f】, Forest normal test score:【%f】' % (score_tree, score_forest))

def cross_val():
    global clf, rfc
    cross_score_tree = cross_val_score(clf, wine_data.data, wine_data.target, cv=10).mean()
    cross_score_forest = cross_val_score(rfc, wine_data.data, wine_data.target, cv=10).mean()
    print('Tree corss_val_test socre:【%f】, Forest cross_val_test score:【%f】' % (cross_score_tree, cross_score_forest))

def forest_grid_search():
    parameters = {
        'n_estimators': [*range(10, 50, 5)],
        'max_depth': [*range(1, 6)],
        'min_samples_leaf': [*range(1, 10, 2)],
        'criterion': ['entropy', 'gini']
    }
    GS = GridSearchCV(rfc, param_grid=parameters, cv=10).fit(wine_data.data, wine_data.target)
    print('Forest best params: %s\n Forest best score:【%f】' % (GS.best_params_, GS.best_score_))

def tree_grid_search():
    parameters = {
        'max_depth': [*range(1, 6)],
        'min_samples_leaf': [*range(1, 10, 2)],
        'criterion': ['entropy', 'gini']
    }
    GS = GridSearchCV(clf, param_grid=parameters, cv=10).fit(wine_data.data, wine_data.target)
    print('Tree best params: %s\nTree best score:【%f】' % (GS.best_params_, GS.best_score_))

normal_compare()
cross_val()
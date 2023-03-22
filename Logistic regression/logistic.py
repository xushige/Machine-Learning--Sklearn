from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## 逻辑回归：尤其适合线性关系的拟合。
## y = sigmoid(wx)

'''数据集'''
data = load_breast_cancer()
x = data['data']
y = data['target']
print(x.shape, y.shape)

'''
逻辑回归模型
penalty: ['l1', 'l2']  default: 'l2'
C: 正则化强度的的倒数, (0, 1)的浮点数, C越小, 正则化强度越大
'''
lr1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000)
lr2 = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
lr1.fit(x, y)
lr2.fit(x, y)
# 查看每个特征的重要性, L1正则化会使得某些特征重要性变为0，相当于降维；而L2只会趋向于0
print(lr1.coef_.shape, lr2.coef_.shape)  

'''切分数据集'''
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, shuffle=True)

'''C值选取'''
l1, l2 = [], []
for i in range(5, 101, 5):
    c = i/100
    lr1 = linear_model.LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=1000)
    lr1.fit(xtrain, ytrain)
    acc1 = (lr1.predict(xtest)==ytest).sum() / len(ytest)
    l1.append(acc1)
    
    lr2 = linear_model.LogisticRegression(penalty='l2', solver='liblinear', C=c, max_iter=1000)
    lr2.fit(xtrain, ytrain)
    acc2 = (lr2.predict(xtest)==ytest).sum() / len(ytest)
    l2.append(acc2)
plt.plot([i/100 for i in range(5, 101, 5)], l1, label='LR-L1')
plt.plot([i/100 for i in range(5, 101, 5)], l2, label='LR-L2')
plt.legend()
plt.xlabel('C value')
plt.title('c value selection')
plt.ylabel('Test, Accuracy')
plt.savefig('C_value_selection')
plt.clf()    
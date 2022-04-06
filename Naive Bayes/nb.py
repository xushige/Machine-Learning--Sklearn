from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
'''数据集载入，标准化，切分'''
data = load_digits()
x, y = data['data'], data['target']
x = StandardScaler().fit_transform(x)
print('data shape: %s     label shape: %s' % (x.shape, y.shape))
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, shuffle=True)

'''朴素贝叶斯建模'''
# prior: y长度为类别数的一维数组形式，表示类别的先验概率，即P(Y=yi)，不指定则自行根据数据计算.默认None
# var_smoothing: 浮点数，让计算平稳。默认1e-9
nb = GaussianNB().fit(trainx, trainy)
# R2-score
score = nb.score(testx, testy)
print('predict R2 score is: %.4f'%(score))
# 概率
prob = nb.predict_proba(testx)
print('probability shape: %s----表示每个样本经过最大后验估计后对应每个label的概率'%str(prob.shape))

'''混淆矩阵可视化'''
cm = confusion_matrix(testy, nb.predict(testx))
print('=======================================\nconfusion matrix:\n%s\n======================================='%(cm))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(i-0.2, j+0.2, str(cm[j][i]), color='white' if cm[j][i]>35 else 'black')
plt.title('confusion matrix')
plt.xticks([*range(prob.shape[1])])
plt.yticks([*range(prob.shape[1])])
plt.xlabel('Pred Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion matrix')
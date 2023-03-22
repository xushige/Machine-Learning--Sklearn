from cgi import test
from sklearn.datasets import make_blobs, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import time

'''SVM二分类可视化'''
def svm_vis(x, y, filename, kernal='linear'):
    print('x_size: %s, y_size:%s'%(x.shape, y.shape))
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='rainbow')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axisx = np.linspace(xlim[0], xlim[1], 30)
    axisy = np.linspace(ylim[0], ylim[1], 30)
    axisx, axisy = np.meshgrid(axisx, axisy)
    xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
    clf = SVC(kernel=kernal).fit(x, y)
    p = clf._decision_function(xy).reshape(axisx.shape)
    ax.contour(axisx, axisy, p, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='black')
    plt.savefig(filename)
    plt.clf()

# 线性数据分类
x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
svm_vis(x, y, 'linear_clf', 'linear') # 线性核可分

# 环形数据分类
x, y = make_circles(n_samples=100, factor=0.1, noise=0.1)
svm_vis(x, y, 'circle_clf_linear', 'linear') # 线性核不可分
svm_vis(x, y, 'circle_clf_rbf', 'rbf') # 使用rbf高斯径向基核处理线性不可分数据


'''乳腺癌数据集svm分类'''
data = load_breast_cancer()
x, y = data['data'], data['target']
x = StandardScaler().fit_transform(x)
print('乳腺癌数据集详情：x_size: %s, y_size:%s'%(x.shape, y.shape))

# 降维可视化数据集分布
pca_xdata = PCA(2).fit_transform(x)
color = ['r', 'b']
plt.scatter(pca_xdata[:, 0][y==0], pca_xdata[:, 1][y==0])
plt.scatter(pca_xdata[:, 0][y==1], pca_xdata[:, 1][y==1])
plt.savefig("breast_dataset_vis")
plt.clf()

# 核函数性能探究
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, shuffle=True)
kernels = ['linear', 'poly', 'rbf', 'sigmoid'] # 线性核，高斯核，sigmoid核
for kernel in kernels:
    start = time.time()
    # cahce-size表示分配多少内存进行计算，越大越快
    # degree：表示多项式核的次幂，degree=1时就是线性核(仅多项式核poly可调)
    # gamma： {'scale', 'auto'} or float, default='scale'。（rbf，poly，sigmoid可调）
    # coef0: float, default=0.0。（poly, sigmoid可调）
    # C值：float, default=1，惩罚系数，C越大，惩罚越大，分类精确
    clf = SVC(kernel=kernel, gamma='auto', cache_size=5000, degree=1) 
    clf.fit(xtrain, ytrain)
    timespan = time.time() - start
    score = clf.score(xtest, ytest)
    print("【%s】 kernel's score is: %f, time-consumption is: %fs"%(kernel, score, timespan))
    '''
    【linear】 kernel's score is: 0.970760, time-consumption is: 0.002428s
    【rbf】 kernel's score is: 0.959064, time-consumption is: 0.003242s
    【sigmoid】 kernel's score is: 0.959064, time-consumption is: 0.002081s
    从结果看，线性核效果较好，高斯核效果差很多，可能是量纲差距过大，或者过拟合
    尝试标准化后高斯核效果与线性核差不多
    【linear】 kernel's score is: 0.976608, time-consumption is: 0.001769s
    【poly】 kernel's score is: 0.959064, time-consumption is: 0.001492s
    【rbf】 kernel's score is: 0.976608, time-consumption is: 0.002371s
    【sigmoid】 kernel's score is: 0.964912, time-consumption is: 0.002023s
    '''
# 对poly核进行网格搜索选择gamma，coef0参数
cv = StratifiedShuffleSplit(5, test_size=0.3)
param_grid = {'gamma': np.logspace(-10, 1, 20), 'coef0':np.linspace(0, 5, 10)}
grid = GridSearchCV(SVC(cache_size=5000, degree=1, kernel='poly'), param_grid=param_grid, cv=cv).fit(x, y)
print('To "poly" kernel, The best parameters are [%s], score is [%f]' % (grid.best_params_, grid.best_score_))

# C值选取曲线
score = []
crange = np.linspace(0.01, 30, 50, dtype=np.float16)
for c in crange:
    svm = SVC(C=c, cache_size=5000, kernel='linear').fit(xtrain, ytrain)
    score.append(svm.score(xtest, ytest))
bestsocre = max(score)
bestc = crange[score.index(bestsocre)]
print('max score: %.4f, c value: %.4f'%(bestsocre, bestc))
plt.plot(crange, score)
plt.scatter([bestc], [bestsocre], c='r', s=40)
plt.title('C value selection')
plt.xlabel('C value')
plt.ylabel('score')
plt.xticks(crange, rotation=-270, fontsize=4)
plt.grid()
plt.rcParams['savefig.dpi'] = 400  # 图片像素
plt.rcParams['figure.dpi'] = 400  # 分辨率
plt.savefig('Cvalue_selection')
plt.clf()

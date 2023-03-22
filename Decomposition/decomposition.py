from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

'''降维算法一定会带来信息损失'''

iris = load_iris()
x = iris['data']
y = iris['target']
print(x.shape, y.shape) # 四维数据

'''n_components: 降到某维'''
pca = PCA(n_components=2)
pca = pca.fit(x)
x_pca = pca.transform(x)
print(x_pca.shape)

'''PCA降维, 可视化数据分布'''
for classes in range(len(set(y))):
    plt.scatter(x_pca[y==classes, 0], x_pca[y==classes, 1], label=iris['target_names'][classes])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.savefig('PCA_visualization')
plt.clf()

'''explained_variance_: 可解释方差, 方差越大, 信息越丰富, 对于分类也越有效'''
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance, explained_variance_ratio)
print('信息保留率：%f' % (explained_variance_ratio.sum()))

'''可视化n_components--explained_variance_ratios曲线, 手肘法选取n值'''
explained_variance_ratios = []
for n in range(1, 5):
    pca = PCA(n)
    pca = pca.fit(x)
    explained_variance_ratios.append(pca.explained_variance_ratio_.sum())
plt.plot([*range(1, 5)], explained_variance_ratios)
plt.xticks([*range(1, 5)])
plt.xlabel('n-components')
plt.ylabel('explained_variance_ratio')
plt.grid(axis='y')
plt.savefig('n-componnets selection')
plt.clf()
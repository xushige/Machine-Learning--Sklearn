from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score # 评估轮廓系数, 检测聚类效果
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

## 聚类思想：簇内差异小，簇外差异大

'''
创建数据集
n_samples: 数据量
n_features: 特征维度
centers: 类别数
'''
x, y =  make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

'''可视化原始数据分布'''
fig, ax1 = plt.subplots(1) # 画布上仅有一个子图
ax1.scatter(x[:, 0], x[:, 1], marker='o')
plt.savefig('data_distribution')
plt.clf()

'''可视化聚类效果'''
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(x)
y_pred = kmeans.labels_ # 训练数据的预测标签
centers = kmeans.cluster_centers_ # 簇中心坐标
for each in range(n_clusters):
    plt.scatter(x[y_pred==each, 0], x[y_pred==each, 1], marker='o', label='class:%d'%each)
plt.legend()
plt.savefig('kmeans_vis')
plt.clf() 

'''轮廓系数'''
score = silhouette_score(x, y_pred)
print('轮廓系数为：【%.4f】'%score)

'''基于轮廓系数选择k值'''
res = []
for k in range(2, 10):
    model = KMeans(k)
    model.fit(x)
    y_pred = model.labels_
    res.append(silhouette_score(x, y_pred))
plt.plot([*range(2, 10)], res, marker='o')
plt.xlabel('K value selection')
plt.ylabel('silhouette_score')
plt.savefig('K_value_selection')
plt.clf()
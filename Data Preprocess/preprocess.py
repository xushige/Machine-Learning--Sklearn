from sklearn.impute import SimpleImputer
import pandas as pd

'''该文件不可运行 仅提供各种预处理方法的使用'''

data = 0
# 缺值填充
# strategy: 按列计算mean, median(中位数), most_frequent(众数), constant进行填充. defalut: mean
# fill_value: constant 可用，表示定值填充. default: None
si = SimpleImputer()
data = si.fit_transform(data) #data.shape==(n, m), n个样本，m维特征

# 分类文字型数据编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le = le.fit(data) # data一维数组
label = le.transform(data)
'''也可以直接label = le.fit_transform(data)'''
data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1]) # 实际对DataFrame最后一列进行编码操作

# 将分类特征转换为分类数值
from sklearn.preprocessing import OrdinalEncoder
data.iloc[:, :] = OrdinalEncoder().fit_transform(data.iloc[:, :]) # 相比LabelEncoder可以编码多列分类数据

# 将分类特征通过OrdinalEncoder直接变成0，1，2忽略了数字本身的关联性
# 因此使用哑变量更能准确刻画并行无关联的分类特征
from sklearn.preprocessing import OneHotEncoder
result = OneHotEncoder(categories='auto').fit_transform(data.iloc[:, :])
# 这里result是对指定列进行读热码转换，一列变为五列，因此需要将原列删除，将读热哑变量进行concat
data.drop(['column1', 'column2'])
data = pd.concat([data, pd.DataFrame(result)], axis=1) # 按列合并
data.columns = ['new_column0', 'new_column1', 'new_column2']

# 连续性特征根据阈值二值化
from sklearn.preprocessing import Binarizer
data.iloc[:, 0:1] = Binarizer(threshold=30).fit_transform(data.iloc[:, 0:1]) #不能使用一维数组
## 通过sklearn工具包内置的CCA实现
import numpy as np
from sklearn.cross_decomposition import CCA
from icecream import ic   # ic用于显示，类似于print

A = [[3, 4, 5, 6, 7] for i in range(2000)]
B = [[8, 9, 10, 11, 12] for i in range(2000)]
# 注意在A、B中的数为输入变量及输出变量参数

# 建模
cca = CCA(n_components=1)  # 若想计算第二主成分对应的相关系数，则令cca = CCA(n_components=2)
# 训练数据
cca.fit(X, Y)
# 降维操作
X_train_r, Y_train_r = cca.transform(X, Y)
#输出相关系数
ic(np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1])  #如果想计算第二主成分对应的相关系数 print(np.corrcoef(X_train_r[:, 1], Y_train_r[:, 1])[0, 1])


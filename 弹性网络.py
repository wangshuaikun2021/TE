from sklearn.linear_model import ElasticNet



# 初始化弹性网络回归器
reg = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False)
# 拟合线性模型
reg.fit(X, y)
# 权重系数
w = reg.coef_
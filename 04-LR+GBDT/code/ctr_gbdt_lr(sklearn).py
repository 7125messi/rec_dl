from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


gbm1 = GradientBoostingClassifier(n_estimators=50, random_state=10, subsample=0.6, max_depth=7,
                                  min_samples_split=900)
gbm1.fit(X_train, Y_train)
train_new_feature = gbm1.apply(X_train)
train_new_feature = train_new_feature.reshape(-1, 50)

enc = OneHotEncoder()

enc.fit(train_new_feature)

# # 每一个属性的最大取值数目
# print('每一个特征的最大取值数目:', enc.n_values_)
# print('所有特征的取值数目总和:', enc.n_values_.sum())

train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())
"""
model.apply(X_train)的用法
model.apply(X_train)返回训练数据X_train在训练好的模型里每棵树中所处的叶子节点的位置（索引）

sklearn.preprocessing 中OneHotEncoder的使用
除了pandas中的 get_dummies()，sklearn也提供了一种对Dataframe做One-hot的方法。

OneHotEncoder() 首先fit() 过待转换的数据后，再次transform() 待转换的数据，就可实现对这些数据的所有特征进行One-hot 操作。
由于transform() 后的数据格式不能直接使用，所以最后需要使用.toarray() 将其转换为我们能够使用的数组结构。
enc.transform(train_new_feature).toarray()

sklearn中的GBDT 能够设置树的个数，每棵树最大叶子节点个数等超参数，但不能指定每颗树的叶子节点数。
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
 
 
##========================= load data ========================================
 
df = pd.read_csv("../data/ctr_data.csv") 

# 本文只是做一个demo，所以不进行特征工程，只选择部分属性
cols = ['C1',
        'banner_pos', 
        'site_domain', 
        'site_id',
        'site_category',
        'app_id',
        'app_category', 
        'device_type', 
        'device_conn_type',
        'C14', 
        'C15',
        'C16']
 
cols_all = ['id']
cols_all.extend(cols)
print(df.head(10))
 
y = df['click']  
y_train = y.iloc[:-2000] # training label
y_test = y.iloc[-2000:]  # testing label
 
X = df[cols_all[1:]]  # training dataset
 
# label encode
lbl = preprocessing.LabelEncoder()
X['site_domain'] = lbl.fit_transform(X['site_domain'].astype(str))#将提示的包含错误数据类型这一列进行转换
X['site_id'] = lbl.fit_transform(X['site_id'].astype(str))
X['site_category'] = lbl.fit_transform(X['site_category'].astype(str))
X['app_id'] = lbl.fit_transform(X['app_id'].astype(str))
X['app_category'] = lbl.fit_transform(X['app_category'].astype(str))
 
X_train = X.iloc[:-2000]
X_test = X.iloc[-2000:]  # testing dataset
 
 
##=========================== gbdt -lightgbm =================================
 
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
 
# number of leaves,will be used in feature transformation
num_leaf = 64
 
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)
# 训练100轮的结果是[100] training's binary_logloss: 0.408675。
 
print('Save model...')
# save model to file
gbm.save_model('../result/model.txt')
 
print('Start predicting...')
# predict and get data on leaves, training data
 
 
##================= convert raw data to sparse-concatenate-new-data ==========
 
##====================== 训练集转换

# 用训练好的lgb模型预测训练集，观察其落在哪些叶子节点上
# 返回训练数据在训练好的模型里预测结果所在的每棵树中叶子节点的位置（索引），形式为8000*100的二维数组。
print('Writing transformed training data')
y_pred_train = gbm.predict(X_train, pred_leaf=True)

print(y_pred_train.shape)
# 共有8000个样本，100棵树（在上面的参数中 num_trees=100),观察第 1 个样本y_pred_train[0]的前10个值
print(y_pred_train[0][:10])
# 其中 第一个数 31 表示这个样本落到了第一颗树的 31 叶子节点，29 表示落到了第二棵树的 29 叶子节点，注意31 、29表示节点编号，从0开始到63。

# 将叶子节点编号转化为OneHot编码
# 构造Ont-hot数组作为新的训练数据
"""
这里并没有使用sklearn中的OneHotEncoder()，也没有使用pandas中的get_dummies()，而是手工创建一个One-hot数组。
首先，创建一个二维零数组用于存放one-hot的元素；
然后，获取第2步得到的二维数组里每个叶子节点在整个GBDT模型里的索引号，因为一共有100棵树，每棵树有64个叶子节点，
所以索引范围是0~6400；
（这里有一个技巧，通过把每棵树的起点索引组成一个列表，
再加上由落在每棵树叶子节点的索引组成的列表，
就得到了往二维零数组里插入元素的索引信息）
最后，
temp = np.arange(len(y_pred_train[0])) * num_leaf + np.array(y_pred_train[i])
"""
transformed_training_matrix = np.zeros([len(y_pred_train), len(y_pred_train[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred_train)):
    temp = np.arange(len(y_pred_train[0])) * num_leaf + np.array(y_pred_train[i])
    # 对二维数组填充信息，采用"+=" 的方法
    transformed_training_matrix[i][temp] += 1
 
##===================== 测试集转换
print('Writing transformed testing data')
y_pred_test = gbm.predict(X_test, pred_leaf=True)
 
transformed_testing_matrix = np.zeros([len(y_pred_test), len(y_pred_test[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred_test)):
    temp = np.arange(len(y_pred_test[0])) * num_leaf + np.array(y_pred_test[i])
    transformed_testing_matrix[i][temp] += 1
 
 
##=================================  LR ======================================
lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
lm.fit(transformed_training_matrix,y_train)  # fitting the data
y_pred_lr_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
 
 
##===============================  metric ====================================
# 在Kaggle指明的评价指标是NE(Normalized Cross-Entropy)
NE = (-1) / len(y_pred_lr_test) * sum(((1+y_test)/2 * np.log(y_pred_lr_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_lr_test[:,1])))
print("Normalized Cross Entropy " + str(NE))
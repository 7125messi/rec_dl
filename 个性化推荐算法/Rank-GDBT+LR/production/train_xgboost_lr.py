import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import xgboost as xgb
from ana_train_test_data import ana_train_test_data
from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib

def train_xgboost_and_lr_model(input_train_data,input_test_data,mix_xgboost_model_file,mix_lr_model_file):
    """
    :param input_train_data:
    :param input_test_data:
    :param mix_xgboost_model_file:
    :param mix_lr_model_file:
    """
    X_train, X_test, y_train, y_test = ana_train_test_data(input_train_data, input_test_data)
    # 保存 DMatrix 到 XGBoost 二进制文件中后, 会在下次加载时更快:
    train_mat = xgb.DMatrix(data=X_train, label=y_train)
    train_mat.save_binary('../data/train_bin')
    tree_num,tree_depth,learning_rate = 50,4,0.3
    # 调用训练树模型的主体函数
    # 混合模型的训练是分开训练：树模型与lr模型分开训练
    best_xgboost_model = train_xgboost_model_core(train_mat,tree_num,tree_depth,learning_rate)
    best_xgboost_model.save_model(mix_xgboost_model_file)

    # lr模型的训练：所需特征是树模型经过编码得到（transformer）
    # 只需要叶子节点编成离散化特征
    # 预测样本落在哪个叶子节点上————>lr
    tree_leaf = best_xgboost_model.predict(train_mat,pred_leaf=True)
    # print(len(tree_leaf)) # 27133
    # print(tree_leaf[0]) # [22 24 23 23 22 25 28 15 15 17]
    # print(np.max(tree_leaf)) # 30
    """
    27133
    
    [22 24 23 23 22 25 28 15 15 17 15 16 16 16 19 15 15 17 15 15 19 15 15 15
     15 13 15 15 17 15 16 17 19 15 11 15 17 15 15 23 13 13 15 15 15 15 15 17
     15 15]
     
    上面是第1个样本最后分别被划分到50颗树的哪个叶子节点上
    这里有2**4*50 = 800维的特征，而我们的训练样本数维27133
    
    一般在，实战中：特征数：样本数=1：100
    
    我们这里的可以将tree_num减少为10颗树
    """

    # 获取lr所需特征
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_depth)
    lr_cf = LogisticRegressionCV(Cs=[1], penalty='l2', tol=0.0001, max_iter=500, cv=5, scoring='roc_auc').fit(total_feature_list,y_train)
    scores = lr_cf.scores_['1']
    print(scores)
    print("Diffent:{}".format(",".join([str(ele) for ele in scores.mean(axis=0)])))
    print("AUC:{0},(+-{1:.2f})".format(scores.mean(), scores.std() * 2))
    """
    [[0.88769946]
     [0.88665395]
     [0.89399759]
     [0.89204707]
     [0.89908811]]
    Diffent:0.891897234262746
    AUC:0.891897234262746,(+-0.01)
    
    这里相比之前的best_xgboost_model auc = 0.8742，提高了不少
    """
    # 模型序列化
    joblib.dump(lr_cf,mix_lr_model_file)


# 提取特征的函数
def get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_depth):
    """
    :param tree_leaf: prediction the tree model
    :param tree_num: total_tree_num
    :param tree_depth: total_tree_depth
    :return: a sparse matrix to record total train feature for lr part of mixed model:
    如果tree_depth=6，叶子节点个数为2的6次方为64，tree_num=10，那么总共有640维特征。而实际中，tree_depth，tree_num可能更大，所以维度更高，所以采用稀疏矩阵来返回
    """
    total_node_num = 2**(tree_depth + 1) - 1
    yezi_node_num = 2**(tree_depth)
    no_yezi_node_num = total_node_num - yezi_node_num
    total_col_num = yezi_node_num * tree_num
    total_row_num = len(tree_leaf)

    col = []
    row = []
    data = []

    base_row_index = 0
    for one_result in tree_leaf:
        base_col_index = 0
        # [22 24 23 23 22 25 28 15 15 17]
        for fix_index in one_result:
            yezi_index = fix_index - no_yezi_node_num
            yezi_index = yezi_index if yezi_index >= 0 else 0
            col.append(base_col_index + yezi_index) # 这里第一颗树预测结果叶子节点编号为0-15（2**4），第二颗树为16-31。。。
            row.append(base_row_index)
            data.append(1)  # 该位置数据为1
            base_col_index += yezi_node_num
        base_row_index += 1
        # 转化为稀疏矩阵
    total_feature_list = coo_matrix((data,(row,col)),shape=(total_row_num,total_col_num))
    return total_feature_list

def train_xgboost_model_core(train_mat,tree_num,tree_depth,learning_rate):
    """
    :param train_mat: train data and label
    :param tree_num:
    :param tree_depth:
    :param learning_rate:step_size
    :return:Booster
    """
    # XGBoost使用pair格式的list来保存参数
    # Booster（提升）参数
    # param = {'bst:max_depth': 2, 'bst:eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    # param['nthread'] = 4
    # param['eval_metric'] = 'auc'
    para_dict = {
        'max_depth':tree_depth,
        'eta':learning_rate,
        'objective': 'reg:linear',
        'silent':1
    }
    base_model = xgb.train(para_dict,train_mat,tree_num)
    return base_model


if __name__ == "__main__":
    train_xgboost_and_lr_model('../data/train.csv','../data/test.csv','../data/xgboost_mix_model','../data/lr_mix_model.pkl')
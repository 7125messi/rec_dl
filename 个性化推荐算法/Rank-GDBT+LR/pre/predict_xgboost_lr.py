import numpy as np
import pandas as pd
import sys
sys.path.append('../production/')
from ana_train_test_data import ana_train_test_data
import xgboost as xgb
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.externals import joblib
from scipy.sparse import coo_matrix



"""
# 在训练完成之后可以将模型保存下来，也可以查看模型内部的结构
bst.save_model('test.model')

# 导出模型和特征映射（Map）
你可以导出模型到txt文件并浏览模型的含义：
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.raw.txt','featmap.txt')

# 加载模型
通过如下方式可以加载模型：
bst = xgb.Booster({'nthread':4}) # init model
bst.load_model("model.bin")      # load data
"""
def pre_xgboost_model(input_train_data,input_test_data,best_xgboost_model):
    X_train,X_test,y_train,y_test = ana_train_test_data(input_train_data,input_test_data)
    dtest = xgb.DMatrix(X_test)
    xgboost_model = xgb.Booster(model_file=best_xgboost_model)

    # XGBoost预测的输出是概率,如果是一个二类分类问题，输出值是样本为第一类的概率，此时我们需要将概率值转换为0或1
    y_pred_prob = xgboost_model.predict(dtest)
    y_pred = [round(value) for value in y_pred_prob]
    accuracy = accuracy_score(y_test.astype('float'),y_pred)
    print("Accuracy:",accuracy)
    # print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

    aucscore = roc_auc_score(y_test,y_pred_prob) # 0.859概率
    print("Auc_score:",aucscore)

def pre_xgboost_and_lr_model(input_train_data,input_test_data,xgboost_mix_model,lr_mix_model):
    X_train,X_test,y_train,y_test = ana_train_test_data(input_train_data,input_test_data)
    test_mat = xgb.DMatrix(data=X_test, label=y_test)
    xgboost_model = xgb.Booster(model_file=xgboost_mix_model)
    tree_leaf = xgboost_model.predict(test_mat, pred_leaf=True)

    tree_num = 50
    tree_depth = 4

    # 获取lr所需特征
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth)
    lr_model = joblib.load(lr_mix_model)

    # 预测标签
    y_pred = lr_model.predict(total_feature_list)
    # 预测属于某标签的概率
    y_pred_prob = lr_model.predict_proba(total_feature_list)

    accuracy = accuracy_score(y_test,y_pred)
    print("Accuracy:",accuracy)

    aucscore = roc_auc_score(y_test, y_pred_prob[:,1]) # 0.859概率
    print("Auc_score:",aucscore)

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




if __name__ == "__main__":
    pre_xgboost_model('../data/train.csv','../data/test.csv','../data/best_xgboost_model')
    pre_xgboost_and_lr_model('../data/train.csv', '../data/test.csv', '../data/xgboost_mix_model', '../data/lr_mix_model.pkl')

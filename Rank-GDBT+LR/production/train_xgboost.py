import xgboost as xgb
from ana_train_test_data import ana_train_test_data
import matplotlib.pyplot as plt


def train_xgboost_model(input_train_data,input_test_data,result_file,xgboost_model_file):
    """
    :param input_train_data:
    :param input_test_data:
    :param result_file:
    :param xgboost_model_file:save the bset model
    """
    X_train, X_test, y_train, y_test = ana_train_test_data(input_train_data, input_test_data)
    # 保存 DMatrix 到 XGBoost 二进制文件中后, 会在下次加载时更快:
    train_mat = xgb.DMatrix(data=X_train,label=y_train)
    train_mat.save_binary('../data/train_bin')

    # grid_search(train_mat,result_file) # 仅在参数调优使用

    # 以下参数是grid_search得到的最优参数，可在result_model.csv中得到
    tree_num = 50
    tree_depth = 4
    learning_rate = 0.3
    best_model = train_xgboost_model_core(train_mat,tree_num,tree_depth,learning_rate)

    # 绘制出特征 importance（重要性）, 可以使用 plot_importance. 该函数需要安装 matplotlib
    # 根据其在输入数组中的索引，特征被自动命名为f0...
    xgb.plot_importance(best_model,max_num_features=10)
    # 输出的 tree（树）会通过 matplotlib 来展示, 使用 plot_tree 指定 target tree（目标树）的序号. 该函数需要 graphviz 和 matplotlib.
    # xgb.plot_tree(best_model, num_trees=28)
    # 当您使用 IPython 时, 你可以使用 to_graphviz 函数, 它可以将 target tree（目标树）转换成 graphviz 实例. graphviz 实例会自动的在 IPython 上呈现.
    # xgb.to_graphviz(bst, num_trees=2)
    plt.show()
    best_model.save_model(xgboost_model_file)

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
    # print(xgb.cv(para_dict,train_mat,tree_num,nfold=5,metrics={'auc'}))
    return base_model

def choose_parameter():
    result_list = []
    tree_depth_list = [4,5,6]
    tree_num_list = [10,50,100]
    learning_rate_list = [0.3,0.5,0.7]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                result_list.append((ele_tree_depth,ele_tree_num,ele_learning_rate))
    return result_list

def grid_search(train_mat,result_file):
    """
    :param:train_mat:train data and train label
    select the best parameter for training model
    """
    para_list = choose_parameter()
    for ele in para_list:
        (tree_depth,tree_num,learning_rate) = ele
        para_dict = {
            'max_depth':tree_depth,
            'eta':learning_rate,
            'objective': 'reg:linear',
            'silent':1
        }
        res = xgb.cv(para_dict, train_mat, tree_num, nfold=5, metrics={'auc'})
        auc_score = res.loc[tree_num-1,['test-auc-mean']].values[0] # 选择最后一行tree_num - 1的结果
        print("tree_depth:{0},tree_num:{1},learning_rate:{2},auc:{3}".format(tree_depth,tree_num,learning_rate,auc_score))
        write_res(result_file,tree_depth,tree_num,learning_rate,auc_score)

def write_res(result_file,tree_depth,tree_num,learning_rate,auc_score):
    with open(result_file,'a+') as fw:
        fw.write(str(tree_depth) +','+ str(tree_num) +','+ str(learning_rate) +','+ str(auc_score) + '\n')


if __name__ == "__main__":
    train_xgboost_model('../data/train.csv','../data/test.csv','../data/result_model.csv','../data/best_xgboost_model')
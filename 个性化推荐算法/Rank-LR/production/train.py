from sklearn.linear_model import LogisticRegressionCV
from ana_train_test_data import ana_train_test_data
from sklearn.externals import joblib

def train_lr_model(X_train,y_train,X_test,y_test,model_coef,model_file):
    """
    :param X_train:train data for lr
    :param y_train:train data label for lr
    :param model_coef:w1,w2,...
    :param model_file:model pkl
    """
    # Cs导数即为正则化参数分别验证1，10，100，l2正则倾向于学小,l1正则倾向于学为0,tol=0.0001参数迭代停止条件,最大迭代次数为500次，cv=5折交叉验证，solver优化方法，默认是拟牛顿法，样本多的话用随机梯度下降法

    # 默认scoring = 'accuracy'
    # lr_cf = LogisticRegressionCV(Cs=[1],penalty='l2',tol=0.0001,max_iter=500,cv=5).fit(X_train,y_train)
    # # lr_cf = LogisticRegressionCV(Cs=[1, 10, 100], penalty='l2', tol=0.0001, max_iter=500, cv=5).fit(X_train, y_train)
    # scores = lr_cf.scores_['1']
    # print(scores) # 5折交叉验证，每一折都对1，10，100正则化参数进行打分
    # print("Diffent:{}".format(",".join([str(ele) for ele in scores.mean(axis=0)])))
    # print("Accuracy:{0},(+-{1:.2f})".format(scores.mean(),scores.std()*2))

    # scoring = 'roc_auc'
    lr_cf = LogisticRegressionCV(Cs=[1], penalty='l2', tol=0.0001, max_iter=500, cv=5,scoring='roc_auc').fit(X_train, y_train)
    scores = lr_cf.scores_['1']
    print(scores)
    print("Diffent:{}".format(",".join([str(ele) for ele in scores.mean(axis=0)])))
    print("AUC:{0},(+-{1:.2f})".format(scores.mean(),scores.std()*2))
    # 通常我们在点击率预估中选择auc更高的，所以这里我们选择正则化参数为1
    """
    the X_train shape is: (27133, 106)
    the y_train shape is: (27133,)
    the X_test shape is: (18089, 106)
    the y_test shape is: (18089,)
    [[0.84285188 0.84285188 0.84303611]
     [0.84540262 0.84540262 0.84540262]
     [0.84574272 0.84574272 0.84574272]
     [0.84518983 0.84518983 0.84518983]
     [0.84463693 0.84463693 0.84463693]]
    Diffent:0.8447647951944403,0.8447647951944403,0.844801641178228
    Accuracy:0.8447770771890363
    [[0.84518599 0.84517793 0.84517646]
     [0.84493966 0.84489787 0.84490032]
     [0.86057678 0.8605873  0.86059048]
     [0.85268713 0.85222455 0.85222455]
     [0.85568391 0.85551806 0.85551904]]
    Diffent:0.8518146954365096,0.8516811415183174,0.8516821687923131
    AUC:0.8517260019157132
    """
    coef = lr_cf.coef_[0]
    print(len(coef))

    with open(model_coef,'w+') as fw:
        fw.write(",".join(str(ele) for ele in coef))

    # 模型序列化
    """
    在用sklearn训练好一个模型之后，如何将它持久化到硬盘中，并随时反序列化回来呢？
    sklearn.external.joblib
    pythoner平时对pickle可能更加熟悉, pickle可以对python的对象进行序列化、反序列化操作.
    joblib工具与pickle不同的是，joblib对sklearn创建的对象支持更加友好.
    对于一个15000维，经过将近70万文本分类数据的训练之后，使用pickle对模型进行序列化大约需要270MB的存储空间， 
    而使用joblib仅需要50MB的空间.
    代码如下:
    import sklearn.externals.joblib as jl
    # 序列化操作
    jl.dump(model,'model.pkl')
    # 反序列化操作
    model = jl.load('model.pkl')
    """
    joblib.dump(lr_cf,model_file)

def run_model(input_train_data,input_test_data,model_coef,model_file):
    """
    :param input_train_data:
    :param input_test_data:
    """
    X_train,X_test,y_train,y_test = ana_train_test_data(input_train_data,input_test_data)
    train_lr_model(X_train,y_train,X_test,y_test,model_coef,model_file)


if __name__ == "__main__":
    run_model('../data/train.csv','../data/test.csv','../data/lr_coef','../data/lr_model_file.pkl')

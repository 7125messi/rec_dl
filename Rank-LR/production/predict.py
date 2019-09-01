from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.externals import joblib
from ana_train_test_data import ana_train_test_data

def pre_test_file(X_test,y_test,model_file):
    lr_model = joblib.load(model_file)
    # 预测标签
    y_pred = lr_model.predict(X_test)
    # 预测属于某标签的概率
    y_pred_prob = lr_model.predict_proba(X_test)

    """
    # 1 分类准确率分数是指所有分类正确的百分比。
    # 分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。
    sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数
    """
    accuracy = accuracy_score(y_test,y_pred)
    print("Accuracy:",accuracy)

    """
    # 2 召回率 =提取出的正确信息条数 /样本中的信息条数。通俗地说，就是所有准确的条目有多少被检索出来了
    sklearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)
    
    参数average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]
    将一个二分类matrics拓展到多分类或多标签问题时，我们可以将数据看成多个二分类问题的集合，每个类都是一个二分类。
    接着，我们可以通过跨多个分类计算每个二分类metrics得分的均值，这在一些情况下很有用。你可以使用average参数来指定。
    
    macro：计算二分类metrics的均值，为每个类给出相同权重的分值。当小类很重要时会出问题，因为该macro-averging方法是对性能的平均。
    另一方面，该方法假设所有分类都是一样重要的，因此macro-averaging方法会对小类的性能影响很大。
    
    weighted:对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。
    
    micro：给出了每个样本类以及它对整个metrics的贡献的pair（sample-weight），而非对整个类的metrics求和，
    它会每个类的metrics上的权重及因子进行求和，来计算整个份额。Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。
    
    samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）
    
    average：average=None将返回一个数组，它包含了每个类的得分.
    """
    recall = recall_score(y_test, y_pred, average=None)
    print("Recall:",recall)

    """
    # 3 roc_auc_score
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y_test,y_pred)
    y_pred即可以是类别，也可以是概率。
    roc_auc_score直接根据真实值和预测值计算auc值，省略计算roc的过程。
    
    ### 真实值和预测值
    import numpy as np
    y_test = np.array([0,0,1,1])
    y_pred1 = np.array([0.3,0.2,0.25,0.7])
    y_pred2 = np.array([0,0,1,0])
    
    ### 性能度量auc
    from sklearn.metrics import roc_auc_score
    
    # 预测值是概率
    auc_score1 = roc_auc_score(y_test,y_pred1)
    print(auc_score1)
    
    # 预测值是类别
    auc_score2 = roc_auc_score(y_test,y_pred2)
    print(auc_score2)
    """

    """
    # roc_auc_score 是 预测得分曲线下的 auc，在计算的时候调用了 auc
    
    两种方法都可以得到同样的结果
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    print(roc_auc_score(y_true, y_scores))
    0.75
    fpr,tpr,thresholds=metrics.roc_curve(y_true,y_scores,pos_label=1)
    print(metrics.auc(fpr,tpr))
    0.75
    
    需要注意的是，roc_auc_score 中不能设置 pos_label，而在 roc_curve中，pos_label的默认设置如下
    classes = np.unique(y_true)
    if (pos_label is None and
        not (array_equal(classes, [0, 1]) or
             array_equal(classes, [-1, 1]) or
             array_equal(classes, [0]) or
             array_equal(classes, [-1]) or
             array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.
        
    也就是说，roc_auc_score 中 pos_label 必须满足以上条件，才能直接使用，否则，需要使用 roc_curve 和auc。
    
    import numpy as np
    from sklearn import metrics
    y = np.array([1, 1, 2, 2])
    pred = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
    0.75
    
    print(metrics.roc_auc_score(y,pred))
    ValueError: Data is not binary and pos_label is not specified
    #pos_label 不符合 roc_curve的默认设置，因此报错，可以修改为
    y=np.array([0,0,1,1])  #np.array([-1,-1,1,1]) 
    print(metrics.roc_auc_score(y,pred))
    0.75
    """
    aucscore = roc_auc_score(y_test, y_pred_prob[:,1]) # 0.859概率
    print("Auc_score:",aucscore)

    # 4 混淆矩阵
    confusionmatrix = confusion_matrix(y_test,y_pred)
    print("Confusionmatrix:",confusionmatrix)


def run_pre(input_train_data,input_test_data,model_file):
    X_train, X_test, y_train, y_test = ana_train_test_data(input_train_data, input_test_data)
    pre_test_file(X_test,y_test,model_file)


if __name__ == "__main__":
    run_pre('../data/train.csv','../data/test.csv','../data/lr_model_file.pkl')

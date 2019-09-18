XGBoost(树模型)连续特征不需要离散化处理
只需要对离散特征进行0-1编码

XGBoost+LR

主要流程步骤：

* 第一步：利用production/ana_train_test_data.py进行数据预处理和特征工程
* 第二步：利用production/train_xgboost.py进行xgboost模型训练和参数调优
* 第三步：利用production/train_xgboost_lr.py进行xgboost和lr的混合模型
* 第四步：利用pre/predict_xgboost_lr.py分别利用xgboost和xgboost和lr混合模型进行预测

也可以尝试使用：lightGBM+LR

LR+GBDT是一种具有stacking思想的模型融合器，所以可以用来解决二分类问题。这个方法出自于Facebook 2014年的论文 Practical Lessons from Predicting Clicks on Ads at Facebook。最广泛的使用场景是CTR预估或者搜索排序。
数据集下载地址：https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

![img](https://img-blog.csdnimg.cn/20190129160739988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h0YmVrZXI=,size_16,color_FFFFFF,t_70)

如图中共有两棵树，x为一条输入样本，遍历两棵树后，x样本分别落到两颗树的叶子节点上，每个叶子节点对应LR一维特征，那么通过遍历树，就得到了该样本对应的所有LR特征。构造的新特征向量是取值0/1的。举例来说：上图有两棵树，左树有三个叶子节点，右树有两个叶子节点，最终的特征即为五维的向量。对于输入x，假设他落在左树第一个节点，编码[1,0,0]，落在右树第二个节点则编码[0,1]，所以整体的编码为[1,0,0,0,1]，这类编码作为特征，输入到LR中进行分类。

简单理解就是将GBDT树的叶子节点作为特征输入LR进行训练。

那么关键就有两点：

一、找到样本落在每棵树叶子节点上的位置

二、将位置信息转换成LR的输入数据

对于第一个问题如果你使用的是scikit-learn，那么可以使用apply方法得到，如果你使用的GBDT原始包（xgboost或lightgbm），则在predict时使用参数pred_leaf = True 得到。

对于第二个问题可以使用get_dummies,OneHotEncoder方法，在这里我们采用自己的方法转换。

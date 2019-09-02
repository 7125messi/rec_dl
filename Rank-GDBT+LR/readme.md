XGBoost(树模型)连续特征不需要离散化处理
只需要对离散特征进行0-1编码

XGBoost+LR

主要流程步骤：

* 第一步：利用production/ana_train_test_data.py进行数据预处理和特征工程
* 第二步：利用production/train_xgboost.py进行xgboost模型训练和参数调优
* 第三步：利用production/train_xgboost_lr.py进行xgboost和lr的混合模型
* 第四步：利用pre/predict_xgboost_lr.py分别利用xgboost和xgboost和lr混合模型进行预测

也可以尝试使用：lightGBM+LR

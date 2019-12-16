from sklearn.ensemble import GradientBoostingClassifier as SGBClassifier
from xgboost import XGBClassifier as XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

class GBDTClassifier(object):
    def __init__(self,name = "sklearn",**parameters):
        assert(name == 'sklearn' or name  == 'xgboost')
        self.__pkg_name = name
        self.__parameters = parameters
        self.__feature_encoder = OneHotEncoder()
        self.__classifier = None
        if self.__pkg_name == 'sklearn':
            self.__make_sklearn_model()
        elif self.__pkg_name == 'xgboost':
            self.__make_xgboost_model()

    def __make_sklearn_model(self):
        estimators = self.__parameters['estimators']
        lrate = self.__parameters['learning_rate']
        depth = self.__parameters['max_depth']
        leaf_bodes = self.__parameters['max_leaf_nodes']
        self.__model = SGBClassifier(n_estimators=estimators,
                                     learning_rate=lrate,
                                     max_depth=depth,
                                     max_leaf_nodes=leaf_bodes,
                                     random_state=0)

    def __make_xgboost_model(self):
        estimators = self.__parameters['estimators']
        lrate = self.__parameters['learning_rate']
        depth = self.__parameters['max_depth']
        leaf_bodes = self.__parameters['max_leaf_nodes']
        self.__model = XGBClassifier(nthread = 4,
                                     gamma = 0,
                                     subsample = 0.9,
                                     n_estimators=estimators,
                                     learning_rate=lrate,
                                     max_depth=depth,
                                     max_leaf_nodes=leaf_bodes,
                                     colsample_bytree=0.5)

    def __apply(self,data):
        assert self.__classifier is not None
        applied_data = self.__classifier.apply(data)
        if self.__pkg_name == 'sklearn':
            applied_data = applied_data[:,:,0]
        return applied_data

    def __fit_onehot_encoder(self,data):
        applied_data = self.__apply(data)
        assert applied_data is not None
        self.__feature_encoder.fit(applied_data)

    def __transform_onehot_feature(self,data):
        applied_data = self.__apply(data)
        encoded_feature = self.__feature_encoder.transform(applied_data).toarray()
        return encoded_feature

    def fit(self,samples,lables,split_rate = 0.8):
        assert samples.shape[0] == lables.shape[0]
        train_count = int(samples.shape[0] * split_rate)

        train_samples = samples[0:train_count]
        test_samples = samples[train_count: ]

        train_labels = lables[0:train_count]
        test_labels = lables[train_count: ]

        self.__classifier = self.__model.fit(train_samples,train_labels)
        test_prob = self.__classifier.predict_proba(test_samples)
        test_prob = [prob[1] for prob in test_prob]

        auc = roc_auc_score(test_labels,test_prob)
        print("gbdt with {0} model,get auc {1:.5}".format(self.__pkg_name,auc))

        self.__fit_onehot_encoder(samples)
        return self.__transform_onehot_feature(samples)

    def predict(self,data):
        return self.__classifier.predict(data)

    def predict_trees(self,data):
        return self.__transform_onehot_feature(data)

class GBDTLRPipeline(object):
    def __init__(self,gb_classifier):
        self.__gb_classifier = gb_classifier
        self.__lr_classifier = None

    def fit(self,samples,lables,split_rate = 0.8):
        tree_encoding_samples = self.__gb_classifier(samples,lables,split_rate)
        self.__lr_train(tree_encoding_samples,lables,split_rate)

    def __lr_train(self,samples,lables,split_rate):
        assert samples.shape[0] == lables.shape[0]
        train_count = int(samples.shape[0] * split_rate)

        train_samples = samples[0:train_count]
        test_samples = samples[train_count:]

        train_labels = lables[0:train_count]
        test_labels = lables[train_count:]

        lr_model = LogisticRegression(random_state=0,solver='lbfgs')
        self.__lr_classifier = lr_model.fit(train_samples,train_labels)
        test_prob = self.__lr_classifier.predict_proba(test_samples)
        test_prob = [prob[1] for prob in test_prob]

        auc = roc_auc_score(test_labels, test_prob)
        print("gbdt with lr_model,get auc {0:.5}".format(auc))

    def predict(self):
        return self.__lr_classifier.predict_proba(self.__gb_classifier.predict_trees(data))

    def predict_proba(self):
        prob = self.__lr_classifier.predict_proba(self.__gb_classifier.predict_trees(data))
        return [p[1] for p in prob]

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    X,Y = load_breast_cancer(return_X_y=True)
    ESTIMATORS = 6
    MAX_DEPTH = 2
    LEARNING_RATE = 0.08
    MAX_LEAF_NODES = 3
    sklearn_gbdt_model = GBDTClassifier('sklearn',
                                        estimators = ESTIMATORS,
                                        max_depth = MAX_DEPTH,
                                        learning_rate = LEARNING_RATE,
                                        max_leaf_nodes = MAX_LEAF_NODES)

    gbdt_lr_pipeline_model = GBDTLRPipeline(sklearn_gbdt_model)
    gbdt_lr_pipeline_model.fit(X,Y,0.8)

    p = gbdt_lr_pipeline_model.predict_proba(X)
    predicted = zip(p,Y)

    xgboost_gbdt_model = GBDTClassifier('xgboost',
                                        estimators=ESTIMATORS,
                                        max_depth=MAX_DEPTH,
                                        learning_rate=LEARNING_RATE,
                                        max_leaf_nodes=MAX_LEAF_NODES)

    xgboost_lr_pipeline_model = GBDTLRPipeline(xgboost_gbdt_model)
    xgboost_lr_pipeline_model.fit(X, Y, 0.8)

    p = xgboost_lr_pipeline_model.predict_proba(X)
    predicted = zip(p, Y)

import tensorflow as tf
import os
from sklearn.metrics import roc_auc_score

def get_feature_column():
    """
    all_features:
    age,workclass,education,education-num,marital-status,occupation,relationship,
    race,sex,capital-gain,capital-loss,hours-per-week,native-country,label
    对于连续性特征本身可以直接放入deep feature
    对于连续性特征离散化后放入wide feature
    对于离散型特征先Hash后放入wide feature,然后再embedding放入deep feature
    :return get wide feature and deep feature
    """
    # 连续型
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education-num')
    capital_gain = tf.feature_column.numeric_column('capital-gain')
    capital_loss = tf.feature_column.numeric_column('capital-loss')
    hours_per_week = tf.feature_column.numeric_column('hours-per-week')

    # 离散型
    workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass',hash_bucket_size=512)
    education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=512)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital-status', hash_bucket_size=512)
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=512)
    relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=512)

    # 对age,capital_gain,capital_loss进行离散化
    age_bucket = tf.feature_column.bucketized_column(age,boundaries=[18,25,30,35,40,45,50,55,60,65])
    gain_bucket = tf.feature_column.bucketized_column(capital_gain,boundaries=[0,1000,2000,3000,10000])
    loss_bucker = tf.feature_column.bucketized_column(capital_loss,boundaries=[0,1000,2000,3000,5000])

    # 构建交叉特征
    cross_column = [
        tf.feature_column.crossed_column([age_bucket,gain_bucket],hash_bucket_size=36),
        tf.feature_column.crossed_column([gain_bucket,loss_bucker],hash_bucket_size=16)
    ]

    # wide feature应该包括：哈希、离散化、交叉
    base_column = [
        workclass,
        education,
        marital_status,
        occupation,
        relationship,
        age_bucket,
        gain_bucket,
        loss_bucker
    ]
    wide_column = base_column + cross_column
    # deep feature应该包括：连续值，哈希值的Embeddding,2**9=512
    deep_column = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.embedding_column(workclass,9),
        tf.feature_column.embedding_column(education,9),
        tf.feature_column.embedding_column(marital_status,9),
        tf.feature_column.embedding_column(occupation,9),
        tf.feature_column.embedding_column(relationship,9)
    ]
    return wide_column,deep_column

def build_model_estimator(wide_column,deep_column,model_floder):
    """
    :param wide_column:wide feature
    :param deep_column: deep feature
    :param model_floder: origin model output folder
    :return: model_estimator,serving_input_fn
    """
    # 隐层维度，4层隐层，隐层节点个数决定参数总个数，deep_column 有50维特征：50*128=6400  128*64=8192 64*32=2048 32*16=512
    # 总共17152维度，大概需要1715200个样本，但是我们只有30000个样本，需要数据重复采样55倍
    model_estimator = tf.estimator.DNNLinearCombinedEstimator(
        head=tf.estimator.BaselineClassifier(n_classes=2),
        model_dir=model_floder,
        linear_feature_columns=wide_column,
        linear_optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.1,
            l2_regularization_strength=1.0
        ),
        dnn_feature_columns=deep_column,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.001
        ),
        dnn_hidden_units=[128,64,32,16]
    )
    feature_column = wide_column + deep_column
    feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
    serving_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return model_estimator,serving_input_fn

# 和官方给的模型几乎一样
# https://github.com/tensorflow/models/blob/master/official/r1/wide_deep/census_dataset.py
def input_fn(data_file,re_time,shuffle,batch_num,predict):
    """
    :param data_file:input data,train_data,test_data
    :param re_time:time to repeat the data file
    :param shuffle:shuffle or not [true or false]
    :param batch_num:随机梯度下降时，多少样本更新一下参数，mini batch
    :param predict:train or test [true or false]
    :return:train_feature,train_label or test feature
    """
    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'label'
    ]

    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                            [0], [0], [0], [''], ['']]

    def parse_csv(value):
        columns = tf.io.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        classes = tf.equal(labels, ' >50K')  # binary classification
        return features, classes

    def parse_csv_predict(value):
        columns = tf.io.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        return features

    data_set = tf.data.TextLineDataset(data_file).skip(1).filter(lambda line:tf.not_equal(tf.strings.regex_full_match(line,".*\?.*"),True))
    if shuffle:
        data_set= data_set.shuffle(buffer_size=30000)
    if predict:
        data_set = data_set.map(parse_csv_predict,num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv,num_parallel_calls=5)
    data_set = data_set.repeat(re_time)
    data_set = data_set.batch(batch_num)
    return data_set


def train_wd_model(model_estimator,train_file,test_file,model_export_folder,serving_input_fn):
    """
    :param model_estimator:wd estimator
    :param train_file:
    :param test_file:
    :param model_export_folder: model for tf serving
    :param serving_input_fn: function for model export
    """
    # total_run = 6
    # for index in range(total_run):
    #     model_estimator.train(input_fn=lambda: input_fn(train_file, 10, True, 100, False))  # 训练
    #     print(model_estimator.evaluate(input_fn=lambda: input_fn(test_file, 1, False, 100, False)))  # 测试
    model_estimator.train(input_fn = lambda:input_fn(train_file,20,True,100,False)) # 训练
    print(model_estimator.evaluate(input_fn = lambda:input_fn(test_file,1,False,100,False))) # 测试
    model_estimator.export_savemodel(model_export_folder,serving_input_fn)

def model_performance_test(model_estimator,test_file):
    test_label = get_test_label(test_file)
    result = model_estimator.predict(input_fn=lambda :input_fn(test_file,1,False,100,True))
    predict_list = []
    for one_res in result:
        if "probabilities" in one_res:
            predict_list.append(one_res["probabilities"][1])
    roc_auc_score(predict_list,test_file)


def get_test_label(test_file):
    if not os.path.exists(test_file):
        return []
    linenum = 0
    test_label_list = []
    with open(test_file) as fp:
        for line in fp:
            if linenum == 0:
                linenum += 1
                continue
            if "?" in line.strip():
                continue
            item = line.strip().split(",")
            label_str = item[-1]
            if label_str == " >50K":
                test_label_list.append(1)
            elif label_str == " <=50K":
                test_label_list.append(0)
            else:
                print("error")
    return test_label_list

def run_main(train_file,test_file,model_floder,model_export_folder):
    """
    :param train_file:
    :param test_file:
    :param model_floder:origin floder to put train_model
    :param model_export_folder: for tf.serving
    """
    wide_column,deep_column = get_feature_column()
    model_estimator,serving_input_fn = build_model_estimator(wide_column,deep_column,model_floder)
    train_wd_model(model_estimator,train_file,test_file,model_export_folder,serving_input_fn)
    model_performance_test(model_estimator,test_file)

if __name__ == "__main__":
    run_main('../data/train.csv','../data/test.csv','../data/wd','../data/wd_export')
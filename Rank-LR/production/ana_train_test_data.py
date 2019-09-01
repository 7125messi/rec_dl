import pandas as pd
import numpy as np
pd.set_option('display.max_columns',30)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

########################### data selection and feature selection ###########################



########################### data selection ###########################
def get_input(input_train_data,input_test_data):
    """
    :param input_train_data:
    :param input_test_data:
    :return: pd.DataFrame data_df
    """
    dtype_dict = {
        'age':np.int32,
        'education-num':np.int32,
        'capital-gain':np.int32,
        'capital-loss':np.int32,
        'hours-per-week':np.int32
    }

    use_list = [i for i in range(15)]
    use_list.remove(2)
    train_data_df = pd.read_csv(input_train_data,sep = ',',header = 0, dtype = dtype_dict, na_values=' ?', usecols=use_list)
    train_data_df.dropna(axis=0,how='any',inplace=True)
    test_data_df = pd.read_csv(input_test_data, sep=',', header=0, dtype=dtype_dict, na_values=' ?', usecols=use_list)
    test_data_df.dropna(axis=0, how='any', inplace=True)
    # 合并测试集和训练集
    data_df = pd.concat([train_data_df,test_data_df],axis=0)
    return data_df

########################### feature selection ###########################
def label_trans(x):
    """
    :param x:each element in fixed column of dataframe
    :return: "0" or "1"
    """
    if x == ' <=50K':
        return "0"
    if x == ' >50K':
        return "1"
    return "0"

# 处理标签值
def process_label_feature(label_feature_str,df_in):
    """
    :param label_feature_str:'label'
    :param df_in:DataFrame
    :return:内存中处理
    """
    df_in[label_feature_str] = df_in[label_feature_str].apply(label_trans)
    return df_in


# 处理离散值特征
def process_dis_feature(df,y_label):
    """
    :param df_train:df
    :param y_label:label column
    :return:内存中处理
    """
    dis_feature_df = df.drop(y_label,axis=1)
    y = df[y_label]
    dis_feature_df = pd.get_dummies(dis_feature_df)
    df = pd.concat([dis_feature_df,y],axis=1)
    return df

# 处理连续型特征归一化
def process_con_feature_normalize(df,label):
    """
    :param df:
    :param label:label
    :return:X,y(np.array())
    """
    df_nor = df.drop(label,axis=1)
    X = StandardScaler().fit_transform(df_nor.values)
    y = df[label].values
    return X,y


# 处理连续型特征离散化 age
def process_con_feature_age(df,continuous_features,label):
    con_feature = df[continuous_features].describe()[3:]
    con_feature_bins = con_feature[label].to_numpy()
    df[label] = pd.cut(x=df[label], bins=con_feature_bins,labels=['age_youth', 'age_youngadult', 'age_middleaged', 'age_senior'])
    label_age_df = pd.get_dummies(df[label])
    df = pd.concat([df,label_age_df],axis=1)
    df.drop(label,axis=1,inplace=True)
    return df

def ana_train_test_data(input_train_data,input_test_data):
    """
    :param input_train_data:
    :param input_test_data:
    :param out_train_data:
    :param out_test_data:
    :return: None (实例化至文件中，无返回值)
    """
    data_df = get_input(input_train_data,input_test_data)
    label_feature_str = 'label'
    label = 'age'
    dis_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country']
    continuous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    data_df = process_label_feature(label_feature_str,data_df)
    data_df = process_dis_feature(data_df,label_feature_str)
    data_df = process_con_feature_age(data_df,continuous_features,label)
    # print(data_df[label_feature_str].value_counts())
    X,y = process_con_feature_normalize(data_df,label_feature_str)
    # 切分好训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
    # print(X_test[:10])
    # print(y_test[:10])
    print("the X_train shape is:",X_train.shape)
    print("the y_train shape is:",y_train.shape)
    print("the X_test shape is:", X_test.shape)
    print("the y_test shape is:", y_test.shape)
    return X_train,X_test,y_train,y_test

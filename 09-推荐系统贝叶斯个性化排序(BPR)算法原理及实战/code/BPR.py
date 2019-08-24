import numpy as np
import tensorflow as tf
import os
import random
from collections import defaultdict

tf.compat.v1.disable_v2_behavior()

"""
数据预处理

首先，我们需要处理一下数据，得到每个用户打分过的电影，同时，还需要得到用户的数量和电影的数量。
"""
def load_data():
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open('../data/u.data','r') as f:
        for line in f.readlines():
            u,i,_,_ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u,max_u_id)
            max_i_id = max(i,max_i_id)


    print("max_u_id:",max_u_id)
    print("max_i_idL",max_i_id)

    return max_u_id,max_i_id,user_ratings

"""
下面我们会对每一个用户u，在user_ratings中随机找到他评分过的一部电影i,保存在user_ratings_test，后面构造训练集和测试集需要用到。
"""
def generate_test(user_ratings):
    """
    对每一个用户u，在user_ratings中随机找到他评分过的一部电影i,保存在user_ratings_test，
    后面构造训练集和测试集需要用到。
    """
    user_test = dict()
    for u,i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u],1)[0]
    return user_test

"""
构建训练数据

我们构造的训练数据是<u,i,j>的三元组，i可以根据刚才生成的用户评分字典得到，j可以利用负采样的思想，认为用户没有看过的电影都是负样本：
"""
def generate_train_batch(user_ratings,user_ratings_test,item_count,batch_size=512):
    """
    构造训练用的三元组
    对于随机抽出的用户u，i可以从user_ratings随机抽出，而j也是从总的电影集中随机抽出，当然j必须保证(u,j)不在user_ratings中

    """
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(),1)[0]
        i = random.sample(user_ratings[u],1)[0]
        while i==user_ratings_test[u]:
            i = random.sample(user_ratings[u],1)[0]

        j = random.randint(1,item_count)
        while j in user_ratings[u]:
            j = random.randint(1,item_count)

        t.append([u,i,j])

    return np.asarray(t)

"""
构造测试数据
同样构造三元组，我们刚才给每个用户单独抽出了一部电影，这个电影作为i，而用户所有没有评分过的电影都是负样本j：
"""
def generate_test_batch(user_ratings,user_ratings_test,item_count):
    """
    对于每个用户u，它的评分电影i是我们在user_ratings_test中随机抽取的，它的j是用户u所有没有评分过的电影集合，
    比如用户u有1000部电影没有评分，那么这里该用户的测试集样本就有1000个
    """
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1,item_count + 1):
            if not(j in user_ratings[u]):
                t.append([u,i,j])
        yield np.asarray(t)

"""
模型构建

首先回忆一下我们需要学习的参数θ，
其实就是用户矩阵W(|U|×k)和物品矩阵H(|I|×k)对应的值，对于我们的模型来说，可以简单理解为由id到embedding的转化，因此有：
"""
def bpr_mf(user_count,item_count,hidden_dim):
    u = tf.compat.v1.placeholder(tf.int32,[None])
    i = tf.compat.v1.placeholder(tf.int32,[None])
    j = tf.compat.v1.placeholder(tf.int32,[None])

    user_emb_w = tf.compat.v1.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                 initializer=tf.compat.v1.random_normal_initializer(0, 0.1))
    item_emb_w = tf.compat.v1.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                 initializer=tf.compat.v1.random_normal_initializer(0, 0.1))

    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)

    """
    回想一下我们要优化的目标，第一部分是ui和uj对应的预测值的评分之差，
    再经由sigmoid变换得到的[0,1]值，我们希望这个值越大越好，对于损失来说，当然是越小越好。因此，计算如下：
    """
    x = tf.reduce_sum(tf.multiply(u_emb,(i_emb-j_emb)),axis = 1)

    mf_auc = tf.reduce_mean(tf.compat.v1.to_float(x>0))

    # 第二部分是我们的正则项，参数就是我们的embedding值，所以正则项计算如下：
    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    # 因此，我们模型整个的优化目标可以写作：
    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.math.log(tf.sigmoid(x)))

    train_op = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(bprloss)
    return u, i, j, mf_auc, bprloss, train_op


user_count,item_count,user_ratings = load_data()
user_ratings_test = generate_test(user_ratings)

with tf.compat.v1.Session() as sess:
    u,i,j,mf_auc,bprloss,train_op = bpr_mf(user_count,item_count,20)
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1,4):
        _batch_bprloss = 0
        for k in range(1,5000):
            uij = generate_train_batch(user_ratings,user_ratings_test,item_count)
            _bprloss,_train_op = sess.run([bprloss,train_op],
                                          feed_dict={u:uij[:,0],i:uij[:,1],j:uij[:,2]})

            _batch_bprloss += _bprloss

        print("epoch:",epoch)
        print("bpr_loss:",_batch_bprloss / k)
        print("_train_op")

        user_count = 0
        _auc_sum = 0.0

        for t_uij in generate_test_batch(user_ratings, user_ratings_test, item_count):
            _auc, _test_bprloss = sess.run([mf_auc, bprloss],
                                              feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]}
                                              )
            user_count += 1
            _auc_sum += _auc
        print("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum / user_count)
        print("")
    variable_names = [v.name for v in tf.compat.v1.trainable_variables()]
    values = sess.run(variable_names)
    for k, v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        print(v)

#  0号用户对这个用户对所有电影的预测评分
session1 = tf.compat.v1.Session()
u1_dim = tf.expand_dims(values[0][0], 0)
u1_all = tf.matmul(u1_dim, values[1],transpose_b=True)
result_1 = session1.run(u1_all)
print (result_1)

print("以下是给用户0的推荐：")
p = np.squeeze(result_1)
p[np.argsort(p)[:-5]] = 0
for index in range(len(p)):
    if p[index] != 0:
        print (index, p[index])
import tensorflow as tf
import time
from sklearn.metrics import roc_auc_score
from data import get_data
import pandas as pd

tf.compat.v1.disable_v2_behavior()

x = tf.compat.v1.placeholder(tf.float32,shape=[None,108])
y = tf.compat.v1.placeholder(tf.float32,shape=[None])


m = 2
learning_rate = 0.3
u = tf.Variable(tf.random.normal([108,m],0.0,0.5),name='u')
w = tf.Variable(tf.random.normal([108,m],0.0,0.5),name='w')

U = tf.matmul(x,u)
p1 = tf.nn.softmax(U)

W = tf.matmul(x,w)
p2 = tf.nn.sigmoid(W)

pred = tf.reduce_sum(input_tensor=tf.multiply(p1,p2),axis=1)

cost1=tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
cost=tf.add_n([cost1])
train_op = tf.compat.v1.train.FtrlOptimizer(learning_rate).minimize(cost)
train_x,train_y,test_x,test_y = get_data()
time_s=time.time()
result = []
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(0, 10000):
        f_dict = {x: train_x, y: train_y}

        _, cost_, predict_ = sess.run([train_op, cost, pred], feed_dict=f_dict)

        auc = roc_auc_score(train_y, predict_)
        time_t = time.time()
        if epoch % 100 == 0:
            f_dict = {x: test_x, y: test_y}
            _, cost_, predict_test = sess.run([train_op, cost, pred], feed_dict=f_dict)
            test_auc = roc_auc_score(test_y, predict_test)
            print("%d %ld cost:%f,train_auc:%f,test_auc:%f" % (epoch, (time_t - time_s), cost_, auc, test_auc))
            result.append([epoch,(time_t - time_s),auc,test_auc])

pd.DataFrame(result,columns=['epoch','time','train_auc','test_auc']).to_csv("../data/mlr_"+str(m)+'.csv')
# mlr-5', 'mlr-10','mlr-15','mlr-20 本文对比了在当前给出的数据集下，m=5，10，15，25 以及lr算法的效果
# 可以看到，lr的效果是最好的，随着m的增加，模型的效果越来越差。当然，这并不能说明mlr效果不如lr好，只是我们的数据实在是太少了
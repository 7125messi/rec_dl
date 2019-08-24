import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

"""
Dice激活函数
这里实现的Dice激活函数没有根据上一步的均值方差来计算这一步的均值方差，而是直接计算了这个batch的均值方差。我们可以根据计算出的均值方差对x进行标准化(代码中被注释掉了)，也可以直接调用batch_normalization来对输入进行标准化。
注意的一点是，alpha也是需要训练的一个参数。
"""

def dice(_x,axis=-1,epsilon=0.0000001,name=''):

    alphas = tf.compat.v1.get_variable('alpha'+name,_x.get_shape()[-1],
                             initializer = tf.compat.v1.constant_initializer(0.0),
                             dtype=tf.float32)

    input_shape = list(_x.get_shape())
    reduction_axes = list(range(len(input_shape)))

    del reduction_axes[axis] # [0]

    broadcast_shape = [1] * len(input_shape)  #[1,1]
    broadcast_shape[axis] = input_shape[axis] # [1 * hidden_unit_size]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes) # [1 * hidden_unit_size]
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape) #[1 * hidden_unit_size]
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_normed = tf.compat.v1.layers.batch_normalization(_x, center=False, scale=False)  # a simple way to use BN to calculate x_p
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x
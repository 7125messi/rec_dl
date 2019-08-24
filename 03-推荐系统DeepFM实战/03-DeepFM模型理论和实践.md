# 1、背景

<a name="PLtro"></a>
## 特征组合的挑战

对于一个基于CTR预估的推荐系统，最重要的是**学习到用户点击行为背后隐含的特征****组合**。**在****不同的推荐场景中，低阶组合特征或者高阶组合特征可能都会对最终的CTR产生影响。**

之前介绍的**因子分解机(Factorization Machines, FM)通过对于每一维特征的隐变量内积来提取特征组合。**最终的结果也非常好。但是，**虽然理论上来讲FM可以对高阶特征组合进行建模，但实际上因为计算复杂度的原因一般都只用到了二阶特征组合。**

那么**对于高阶的特征组合来说**，我们很自然的想法，**通过多层的神经网络即DNN去解决。**

<a name="Z2pws"></a>
## DNN的局限
**<br />下面的图片来自于张俊林教授在AI大会上所使用的PPT。

我们之前也介绍过了，对于离散特征的处理，我们使用的是**将特征转换成为one-hot的形式，但是将One-hot类型的特征输入到DNN中，会导致网络参数太多**：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860102918-e55aed6b-98c8-4fa8-870b-856ed25fe58f.webp#align=left&display=inline&height=364&originHeight=364&originWidth=1000&size=0&status=done&width=1000)<br />如何解决这个问题呢，类似于**FFM中的思想，将特征分为不同的field**：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860102897-bb05537f-643a-4068-ab35-6973c17793dd.webp#align=left&display=inline&height=421&originHeight=421&originWidth=1000&size=0&status=done&width=1000)<br />**再加两层的全链接层，让Dense Vector进行组合，那么高阶特征的组合就出来了**<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860102912-0b3ef43c-bf2e-4621-a614-0022b3f0380d.webp#align=left&display=inline&height=451&originHeight=451&originWidth=1000&size=0&status=done&width=1000)<br />**但是低阶和高阶特征组合隐含地体现在隐藏层中，如果我们希望把****低阶特征组合单独建模，然后融合高阶特征组合****。**<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103071-c89efc6c-39c9-4a5f-9957-47160ad9469a.webp#align=left&display=inline&height=386&originHeight=386&originWidth=1000&size=0&status=done&width=1000)<br />即将**DNN与FM进行一个合理的融合**：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103072-56a2f528-3dd4-42e4-a3e4-f3bb6574a095.webp#align=left&display=inline&height=389&originHeight=389&originWidth=1000&size=0&status=done&width=1000)<br />二者的融合总的来说有两种形式，**一是串行结构，二是并行结构**<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103275-564697f3-a164-4cc0-a236-07b17e48fa64.webp#align=left&display=inline&height=488&originHeight=488&originWidth=1000&size=0&status=done&width=1000)<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103224-d9df6d51-d6ff-4ed5-b4d9-030afcb15d37.webp#align=left&display=inline&height=550&originHeight=550&originWidth=1000&size=0&status=done&width=1000)<br />而我们今天要讲到的**DeepFM，就是并行结构中的一种典型代表。<br />**
<a name="0pSWh"></a>
# 2、DeepFM模型

论文：[https://arxiv.org/pdf/1703.04247.pdf](https://arxiv.org/pdf/1703.04247.pdf)

我们先来看一下DeepFM的模型结构：

![image.png](https://cdn.nlark.com/yuque/0/2019/png/200056/1565861334882-d1c70175-f0c6-4dd2-99da-dcb37273e127.png#align=left&display=inline&height=237&name=image.png&originHeight=289&originWidth=570&size=103733&status=done&width=468)

DeepFM包含两部分：**神经网络部分与因子分解机部分**，分别负责**低阶特征的提取和高阶特征的提取**。这两部分**共享同样的输入**。DeepFM的预测结果可以写为：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103398-b0c3fbe8-e1e5-4864-9ed8-1831f4795254.webp#align=left&display=inline&height=100&originHeight=100&originWidth=646&size=0&status=done&width=646)
<a name="1M5DA"></a>
## FM部分

FM部分的详细结构如下：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103588-ad33dc69-c3fc-40b9-9c72-b368e6287dde.webp#align=left&display=inline&height=261&originHeight=324&originWidth=582&size=0&status=done&width=469)<br />FM部分是一个因子分解机。**因为引入了隐变量的原因，对于几乎不出现或者很少出现的隐变量，FM也可以很好的学习。**

FM的输出公式为：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103594-b7606e24-c69d-4f3c-b571-edf4d9c4227e.webp#align=left&display=inline&height=110&originHeight=110&originWidth=558&size=0&status=done&width=558)
<a name="aRlhi"></a>
## 深度部分
![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103596-bbd59df7-d9a9-43c0-996a-d5f2d1f22a40.webp#align=left&display=inline&height=315&originHeight=315&originWidth=538&size=0&status=done&width=538)<br />深度部分是一个**前馈神经网络**。与图像或者语音这类输入不同，图像语音的输入一般是**连续而且密集的**，然而**用于CTR的输入一般是及其稀疏的**。因此**需要重新设计网络结构**。具体实现中为：**在第一层隐含层之前，引入一个嵌入层来完成将输入向量压缩到低维稠密向量。（Embedding Layer）**<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103746-60397bc5-f874-419e-b7d0-7d3a52ac226b.webp#align=left&display=inline&height=149&originHeight=149&originWidth=467&size=0&status=done&width=467)

嵌入层(embedding layer)的结构如上图所示。当前网络结构有两个有趣的特性:

- **1) 尽管不同field的输入长度不同，但是embedding之后向量的长度均为K。**
- **2) ****在FM里得到的隐变量Vik现在作为了嵌入层网络的权重****。**

这里的第二点如何理解呢，假设我们的k=5，首先，对于输入的一条记录，同一个field 只有一个位置是1，那么在由输入得到dense vector的过程中，输入层只有一个神经元起作用，得到的dense vector其实就是输入层到embedding层该神经元相连的五条线的权重，即vi1，vi2，vi3，vi4，vi5。这五个值组合起来就是我们在FM中所提到的Vi。在FM部分和DNN部分，这一块是共享权重的，对同一个特征来说，得到的Vi是相同的。<br />有关模型具体如何操作，我们可以通过代码来进一步加深认识。

<a name="qkUuu"></a>
# 3、相关知识

我们先来讲两个代码中会用到的相关知识吧，代码是参考的github上星数最多的DeepFM实现代码。

<a name="KDFVn"></a>
## Gini Normalization

代码中**将CTR预估问题设定为一个二分类问题，绘制了Gini Normalization来评价不同模型的效果。**

Gini Normalization是什么？？？<br />假设我们有下面两组结果，分别表示**预测值和实际值**：
```
predictions = [0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]
actual = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**然后我们将预测值按照从小到大排列，并根据索引序对实际值进行排序：**
```
Sorted Actual Values [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
```

然后，我们可以画出如下的图片：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103748-1b6eb2a9-4628-4b57-bda2-6af70110c7cc.webp#align=left&display=inline&height=613&originHeight=613&originWidth=903&size=0&status=done&width=903)

接下来我们将**数据Normalization到0，1之间。并画出45度线。**<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103918-b0d823f8-c0c3-43ec-af15-2629c311fbd5.webp#align=left&display=inline&height=609&originHeight=609&originWidth=908&size=0&status=done&width=908)<br />**橙色区域的面积**，就是我们得到的**Normalization的Gini系数**。<br />这里，由于我们是将**预测概率从小到大排的，所以我们希望实际值中的0尽可能出现在前面，因此Normalization的Gini系数越大，分类效果越好。**

<a name="AKDMz"></a>
## embedding_lookup

在tensorflow中有个**embedding_lookup**函数，我们可以**直接根据一个序号来得到一个词或者一个特征的embedding值**，那么他内部其实是包含一个网络结构的，如下图所示：<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565860103906-7b281e60-850a-4795-b078-85513b4a2a54.webp#align=left&display=inline&height=1000&originHeight=1000&originWidth=978&size=0&status=done&width=978)<br />假设我们想要找到**2的embedding值**，这个值其实是**输入层第二个神经元与embedding层连线的权重值。**<br />之前有大佬跟我探讨word2vec输入的问题，现在也算是有个比较明确的答案，输入其实就是**one-hot Embedding，而word2vec要学习的是new Embedding。**


-------------------------------------------------------
-------------------------------------------------------
-------------------------------------------------------
# 1 什么是embedding？

先来看看什么是embedding，我们可以简单的理解为，**将一个特征转换为一个向量**。在推荐系统当中，我们经常会遇到离散特征，如userid、itemid。对于离散特征，我们一般的做法是将其转换为one-hot，但对于itemid这种离散特征，转换成one-hot之后维度非常高，但里面只有一个是1，其余都为0。这种情况下，我们的通常做法就是将其转换为embedding。多值离散特征处理。

embedding的过程是什么样子的呢？它其实就是一层全连接的神经网络，如下图所示：

![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565603006474-19a11a02-6b86-48dc-813d-b49cef48c14b.webp#align=left&display=inline&height=940&originHeight=940&originWidth=1000&size=0&status=done&width=1000)假设一个特征共有5个取值，也就是说one-hot之后会变成5维，我们想将其转换为embedding表示，其实就是接入了一层全连接神经网络。由于只有一个位置是1，其余位置是0，因此得到的embedding就是与其相连的图中红线上的权重。

<a name="FGShJ"></a>
# 2 tf2.0中embedding实现

在tf2.0中，embedding可以通过embedding_lookup来实现，不过不同的是，我们不需要通过sess.run来获取结果了，可以直接运行结果，并转换为numpy。

```python
import tensorflow as tf

embedding = tf.constant(
    [
        [0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]
    ],
    dtype=tf.float32
)

feature_batch = tf.constant([2,3,1,0])
get_embedding1 = tf.nn.embedding_lookup(embedding,feature_batch)
feature_batch_one_hot = tf.one_hot(feature_batch,depth=4)
get_embedding2 = tf.matmul(feature_batch_one_hot,embedding)
print(get_embedding1.numpy().tolist())
```

```python
[
 [0.23000000417232513, 
  0.4300000071525574, 
  0.5299999713897705, 
  0.12999999523162842], 
 [0.23999999463558197, 
  0.4399999976158142, 
  0.5400000214576721, 
  0.14000000059604645], 
 [0.2199999988079071, 
  0.41999998688697815, 
  0.5199999809265137, 
  0.11999999731779099], 
 [0.20999999344348907, 
  0.4099999964237213, 
	0.5099999904632568, 
	0.10999999940395355]
]
```

如果想要在神经网络中使用embedding层，推荐使用Keras：

```python
num_classes=10
input_x = tf.keras.Input(shape=(None,),)
embedding_x = layers.Embedding(num_classes, 10)(input_x)
hidden1 = layers.Dense(50,activation='relu')(embedding_x)
output = layers.Dense(2,activation='softmax')(hidden1)

x_train = [2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7]
y_train = [0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1]

model2 = tf.keras.Model(inputs = input_x,outputs = output)
model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model2.fit(x_train, y_train, batch_size=4, epochs=1000, verbose=0)
```
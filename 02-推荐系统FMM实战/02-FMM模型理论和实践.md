# 1、FFM理论

在CTR预估中，经常会遇到one-hot类型的变量，one-hot类型变量会导致严重的数据特征稀疏的情况，为了解决这一问题，在上一讲中，我们介绍了FM算法。这一讲我们介绍一种在FM基础上发展出来的算法-FFM（Field-aware Factorization Machine）。

**FFM模型中引入了类别的概念，即field**。还是拿上一讲中的数据来讲，先看下图：

![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565854562400-7dc6a5d7-40c3-4f49-8d55-091905d1003a.webp#align=left&display=inline&height=124&originHeight=124&originWidth=275&size=0&status=done&width=275)

在上面的广告点击案例中，“Day=26/11/15”、“Day=1/7/14”、“Day=19/2/15”这三个特征都是代表日期的，可以放到同一个field中。同理，Country也可以放到一个field中。简单来说，**同一个categorical特征经过One-Hot编码生成的数值特征都可以放到同一个field**，包括**用户国籍，广告类型，日期**等等。

在FFM中，每一维特征 xi，针对其它特征的每一种field fj，都会学习一个隐向量 v_i,fj。因此，**隐向量不仅与特征相关，也与field相关。**也就是说，**“Day=26/11/15”这个特征与“Country”特征和“Ad_type"特征进行关联的时候使用不同的隐向量，这与“Country”和“Ad_type”的内在差异相符，也是FFM中“field-aware”的由来。**

假设样本的 n个特征属于 f个field，那么FFM的二次项有 nf个隐向量。而**在FM模型中，****每一维特征的隐向量只有一个**。**FM可以看作FFM的特例，是把所有特征都归属到一个field时的FFM模型**。根据FFM的field敏感特性，可以导出其模型方程。<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565854562659-c71d34f5-7832-4e54-9efa-cd46542f0c23.webp#align=left&display=inline&height=142&originHeight=142&originWidth=764&size=0&status=done&width=764)<br />可以看到，**如果隐向量的长度为 k，那么FFM的二次参数有 nfk 个，远多于FM模型的 nk个。**此外，**由于隐向量与field相关，FFM二次项并不能够化简，其预测复杂度是 O(kn^2)。**

下面以一个例子**简单说明FFM的特征组合方式**。输入记录如下：

![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565854562612-36a2bce9-3225-4261-a28e-3d21845f1e6f.webp#align=left&display=inline&height=146&originHeight=146&originWidth=570&size=0&status=done&width=570)

这条记录可以编码成5个特征，其中**“Genre=Comedy”和“Genre=Drama”属于同一个field**，“Price”是数值型，不用One-Hot编码转换。为了方便说明FFM的样本格式，我们.。

![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565854562769-9515a2da-fc58-4034-95bd-f240954a5870.webp#align=left&display=inline&height=398&originHeight=398&originWidth=784&size=0&status=done&width=784)<br />那么，FFM的组合特征有10项，如下图所示。<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565854562931-d0ca9c18-0299-4e71-818f-353402fa6d7f.webp#align=left&display=inline&height=196&originHeight=196&originWidth=1000&size=0&status=done&width=1000)<br />**其中，红色是field编号，蓝色是feature编号。**

<a name="8RQFG"></a>
# 2、FFM实现细节

这里讲得只是一种FFM的实现方式，并不是唯一的。

**损失函数**<br />FFM将问题定义为分类问题，使用的是logistic loss，同时加入了正则项<br />![](https://cdn.nlark.com/yuque/0/2019/webp/200056/1565854562965-a9ab96fe-3440-4413-94b8-eb0396b28cdd.webp#align=left&display=inline&height=172&originHeight=172&originWidth=716&size=0&status=done&width=716)

什么，这是logisitc loss？第一眼看到我是懵逼的，逻辑回归的损失函数我很熟悉啊，不是长这样的啊？其实是我目光太短浅了。逻辑回归其实是有两种表述方式的损失函数的，取决于你将类别定义为0和1还是1和-1。<br />logistic回归两种形式：<br />第一种形式：label取值为0或1<br />![](https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1565858232984-f8c6e649-62d7-49ff-b0a9-9fa72ced1a7c.jpeg#align=left&display=inline&height=124&originHeight=124&originWidth=514&size=0&status=done&width=514)<br />
<br />第二种形式：将label和预测函数放在一起，label取值为1或-1<br />![](https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1565858233126-a5380ef6-8466-4020-972e-847c23df822d.jpeg#align=left&display=inline&height=68&originHeight=68&originWidth=331&size=0&status=done&width=331)<br />显然，![](https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1565858233115-f655fec0-fac4-45fc-aae6-19f71996c876.jpeg#align=left&display=inline&height=30&originHeight=30&originWidth=321&size=0&status=done&width=321)，上述两种形式等价。<br />
<br />第一种形式的分类法则：<br />![](https://cdn.nlark.com/yuque/0/2019/png/200056/1565858233178-873c8d7e-1f4b-4139-865f-f78c844b28db.png#align=left&display=inline&height=153&originHeight=153&originWidth=291&size=0&status=done&width=291)<br />
<br />第二种形式的分类法则：<br />![](https://cdn.nlark.com/yuque/0/2019/png/200056/1565858233294-ca61056a-28ec-4981-8337-a56d1b0aa3d5.png#align=left&display=inline&height=204&originHeight=204&originWidth=311&size=0&status=done&width=311)<br />
<br />
<br />第一种形式的损失函数可由极大似然估计推出，对于第二种形式的损失函数,<br />![](https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1565858233244-836384b6-9711-4d3b-b4bb-cd3f598d4fdf.jpeg#align=left&display=inline&height=53&originHeight=53&originWidth=492&size=0&status=done&width=492) <br />
<br />左式将分数倒过来，负号提出来，就得到常见的对数损失函数的形式<br />其中，![](https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1565858233408-49171b73-30b3-4b05-bcda-a59e7151e55f.jpeg#align=left&display=inline&height=40&originHeight=40&originWidth=133&size=0&status=done&width=133)<br />
<br />则loss最小化可表示为：<br />![](https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1565858233467-6362a7fb-ccae-41ea-94bb-dd5b94deaccb.jpeg#align=left&display=inline&height=105&originHeight=105&originWidth=530&size=0&status=done&width=530)<br />
<br />上式最后即为极大似然估计的表示形式，则logistic回归模型使用的loss函数为对数损失函数，使用极大似然估计的目的是为了使loss函数最小。
```python
tf_upgrade_v2 --infile "1.x的代码文件" -outfile "2.x的代码文件"
```

```python
tf.compat.v1.disable_v2_behavior()
```

TensorFlow 是谷歌在 2015 年开源的一个通用高性能计算库。从一开始，TensorFlow 的主要目的就是为构建神经网络（NN）提供高性能 API。然而，借助于机器学习（ML）社区对它的兴趣以及时间上的优势，这个类库演变成了一个完整的 ML 生态系统。

TensorFlow凭借自己的性能、易用、配套资源丰富，一举成为当今最炙手可热的AI框架之一。使得目前好多前沿技术、企业项目都用其进行开发。

然而目前，该类库正在经历着从推出以来最大规模的变化。TensorFLow 2.0 已经推出 beta 版本，同 TensorFlow 1.x 版本相比它带来了太多的改变。最大的问题在于不兼容了好多TensorFlow 1.x 版本的API。

这不禁令已有的TensorFlow 1.x用户感到困惑和无重下手。一般来讲，他们大量的工作和成熟代码都是基于TensorFlow 1.x 版本搭建的。面对版本不能兼容的问题，该如何去做？

适配与选择版本问题，再一次站在我们面前，一个不可回避的问题必须要进行解决。今天就为大家分享一下，我们在处理方面的经验，希望对你有所帮助。

# 一、新项目的版本选择
虽然TensorFlow的2.0版本中，有很多光鲜靓丽的新功能。但是TensorFlow 1.x目前比较稳定，建议读者使用TensorFlow 1.x版本开发实际项目，并跟进2.x版本所更新的技术。待2.x版本迭代到2.3以上，再考虑使用2.x版本开发实际项目。

同时开发新项目时，尽量使用动态图+tf.keras接口进行。这样，在以后的移植过程中，可以减少很多不兼容的问题。

如果选择1.x版本进行开发时，尽量使用TensorFlow 1.13.1、1.14版本为主。因为TensorFlow 2.x版本的代码是基于TensorFlow 1.13.1转化而来。TensorFlow 1.13.1版本可以部分支持TensorFlow 2.0版本的代码。而1.14版本在1.13基础上又更新了一代，相对更为稳定。

# 二、TensorFlow 1.x版本与2.x版本共存的解决方案
由于TensorFlow框架的1.x版本与2.x版本差异较大。在1.x版本上实现的项目，有些并不能直接运行在2.x版本上。而新开发的项目推荐使用2.x版本。这就需要解决1.x版本与2.x版本共存的问题。

如用Anaconda软件创建虚环境的方法，则可以在同一个主机上安装不同版本的TensorFlow。

# 三、2.x版本对于静态图的影响

“静态图”是TensorFlow 1.x版本中张量流的主要运行方式。**其运行机制是将“定义”与“运行”相分离**。相当于：**先用程序搭建起一个结构（即在内存中构建一个图），让数据（张量流）按照图中的结构顺序进行计算，最终运行出结果。**

虽然在**TensorFlow 2.x版本中默认的是动态图，但是也可以使用静态图。**

**在TensorFlow 2.x版本中，使用静态图的步骤与在TensorFlow 1.x版本中使用静态图的步骤完全一致。**

但是，**由于静态图不是TensorFlow 2.x版本中的默认工作模式，所以在使用时还需要注意两点：**

- （1）在代码的最开始处，用tf.compat.v1.disable_v2_behavior函数关闭动态图模式。

- （2）将TensorFlow 1.x版本中的静态图接口，替换成tf.compat.v1模块下的对应接口。

例如：
- 将函数tf.placeholder替换成函数tf.compat.v1.placeholder。
- 将函数tf.session替换成函数tf.compat.v1.session。

# 四、将1.x的动态图代码升级到2.x版本

**在TensorFlow 2.x版本中，已经将动态图设为了默认的工作模式。使用动态图时，直接编写代码即可。**

**TensorFlow 1.x中的tf.enable_eager_execution函数在TensorFlow 2.x版本中已经被删除**，另外在TensorFlow 2.x版本中还提供了关闭动态图与启用动态图的两个函数。
- 关闭动态图函数：`tf.compat.v1.disable_v2_behavior`
- 启用动态图函数：`tf.compat.v1.enable_v2_behavior`

# 五、2.x版本中的反向传播

在1.x版本中。动态图的反向传播函数有多个：tf.GradientTape、tfe.implicit_gradients、tfe.implicit_value_and_gradients。可以根据实际的需要来灵活选择，使用起来非常灵活。（具体区别和实例演示可以参考《深度学习之TensorFlow工程化项目实战》一书）

但在2.x中，只保留了tf.GradientTape函数用于计算梯度。tfe.implicit_gradients与tfe.implicit_value_and_gradients函数在TensorFlow 2.x中将不再被支持。

# 六、2.x版本对于估算器的影响
TensorFlow 2.x版本可以完全兼容TensorFlow 1.x版本的估算器框架代码。用估算器框架开发模型代码，不需要考虑版本移植的问题。

# 七、用工具进行代码的版本升级——适用于原生的API代码
如果手里的1.x代码，只使用了原生的API，那么可以直接使用TensorFlow 2.x版本中提供的工具，对TensorFlow 1.x版本的代码进行升级。

在TensorFlow 2.x版本中，提供了一个升级TensorFlow 1.x版本代码的工具——tf_upgrade_v2。该工具可以非常方便地将TensorFlow 1.x版本中编写的代码移植到TensorFlow 2.x中。具体命令如下：
```python
tf_upgrade_v2 --infile "1.x的代码文件" -outfile "2.x的代码文件"
```
该命令主要是个名字匹配，实现了在TensorFlow 2.x版本中，将TensorFlow 1.x版本中的部分函数名字进行调整，部分例子如下：
- 将函数tf.random_uniform 改成了tf.random.uniform。
- 将函数tf.random_crop改成了tf.image.random_crop。
- 将函数tf.random_shuffle改成了tf.random.shuffle。
- 将函数tf.read_file改成了tf.io.read_file。

tf_upgrade_v2工具支持单文件转换和多文件批量转换两种方式。

## 1. 对单个代码文件进行转换

在命令行里输入tf_upgrade_v2命令，用“--infile”参数来指定输入文件，用“--outfile”参数来指定输出文件。具体命令如下：
```python
tf_upgrade_v2 --infile foo_v1.py  --outfile foo_v2.py
```
该命令可以将TensorFlow 1.x版本中编写的代码文件foo_v1.py转成可以支持TensorFlow 2.x版本的代码foo_v2.py。

## 2. 批量转化多个代码文件

在命令行里输入tf_upgrade_v2命令，用“-intree”参数来指定输入文件路径，用“-outtree”参数来指定输出文件路径。具体命令如下：
```python
tf_upgrade_v2 -intree foo_v1  -outtree foo_v2
```
该命令可以将目录为foo_v1下的所有代码文件转成支持TensorFlow 2.x版本的代码文件，并保存到目录foo_v2中。

# 八、2.x版本对于TF-Hub、T2T等库的影响
非常庆幸的是TF-Hub、T2T等库可以支持TensorFlow的1.x与2.x版本。

## 1、TF-Hub库

TF-Hub库是TensorFlow中专门用于预训练模型的库，其中包含很多在大型数据集上训练好的模型。如需在较小的数据集上实现识别任务，则可以通过微调这些预训练模型来实现。另外，它还能够提升原有模型在具体场景中的泛化能力，加快训练的速度。

在GitHub网站上还有TF-Hub库的源码链接，其中包含了众多详细的说明文档。地址如下：

https://github.com/tensorflow/hub

## 2、T2T

Tensor2Tensor（T2T）是谷歌开源的一个模块化深度学习框架，其中包含当前各个领域中最先进的模型，以及训练模型时常用到的数据集。

如想了解更多关于T2T的细节，可以在以下链接中查看T2T框架的源码及教程：

https://github.com/tensorflow/tensor2tensor

## 3、更多实例

在《深度学习之TensorFlow工程化项目实战》一书中，还提供了一个使用TF-Hub库进行微调模型实现分辨男女的例子。以及更多关于T2T的例子。

# 九、2.x版本对于tf.layers接口的影响
用tf.layers接口开发模型代码，需要考虑版本移植的问题。**在TensorFlow 2.x版本中，所有tf.layers接口都需要被换作tf.compat.v1.layers**。

另外，在TensorFlow 2.x版本中，tf.layers模块更多用于tf.keras接口的底层实现。如果是开发新项目，则建议直接使用tf.keras接口。如果要重构已有的项目，也建议使用tf.keras接口进行替换。

# 十、2.x版本的新特性——自动图

**在2.x版本中，加入了很多新特性。自动图是最为实用的特性之一。**

在TensorFlow 1.x版本中，要开发基于张量控制流的程序，必须使用tf.conf、tf. while_loop之类的专用函数。这增加了开发的复杂度。

**在TensorFlow 2.x版本中，可以通过自动图（AutoGraph）功能，将普通的Python控制流语句转成基于张量的运算图。这大大简化了开发工作。**

**在TensorFlow 2.x版本中，可以用tf.function装饰器修饰Python函数，将其自动转化成张量运算图。**示例代码如下：
```python
import tensorflow as tf          			#导入TensorFlow2.0
@tf.function
def autograph(input_data):				#用自动图修饰的函数
    if tf.reduce_mean(input_data) > 0:
      return input_data        			#返回是整数类型
    else:
      return input_data // 2   			#返回整数类型
a =autograph(tf.constant([-6, 4]))
b =autograph(tf.constant([6, -4]))
print(a.numpy(),b.numpy())				#在TensorFlow 2.x上运行，输出:[-3  2] [ 6 -4]
```
从上面代码的输出结果中可以看到，程序运行了控制流“`tf.reduce_mean(input_data) > 0`”语句的两个分支。这表明被装饰器`tf.function`修饰的函数具有张量图的控制流功能。

在使用自动图功能时，如果在被修饰的函数中有多个返回分支，则必须确保所有的分支都返回相同类型的张量，否则会报错。

还有更多新特性，比如TensorFLow.js、TF-Lite、模型保存和恢复的新API等都可以使AI的开发和应用变得更加快捷，方便。具体可以参考《深度学习之TensorFlow工程化项目实战》一书的介绍和实例演示。

# 十一、将代码升级到TensorFlow 2.x版本的经验总结

下面将升级代码到TensorFlow 2.x版本的方法汇总起来，有如下几点。

## 1. 最快速转化的方法

在代码中没有使用contrib模块的情况下，可以在代码最前端加上如下两句，直接可以实现的代码升级。
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
这种方法只是保证代码在TensorFlow 2.x版本上能够运行，并不能发挥TensorFlow的最大性能。

## 2. 使用工具进行转化的方法

在代码中没有使用contrib模块的情况下，用`tf_upgrade_v2`工具可以快速实现代码升级。当然`tf_upgrade_v2`工具并不是万能的，它只能实现基本的API升级。一般在转化完成之后还需要手动二次修改。

## 3. 将静态图改成动态图的方法

静态图可以看作程序的运行框架，可以将输入输出部分原样的套用在函数的调用框架中。具体步骤如下：

（1）将会话（session）转化成函数。

（2）将注入机制中的占位符（tf.placeholder）和字典（feed_dict）转化成函数的输入参数。

（3）将会话运行（session.run）后的结果转化成函数的返回值。

在实现过程中，可以通过自动图功能，用简单的函数逻辑替换静态图的运算结构。自

## 4. 将共享变量的作用于转成Python对象的命名空间

**在定义权重参数时，用tf.Variable函数替换tf.get_variable函数。每个变量的命名空间（variable_scope）用类对象空间进行替换，即将网络封装成类的形式来搭建模型。**

**在封装类的过程中，可以继承tf.keras接口（如：tf.keras.layers.Layer、tf.keras.Model）也可以继承更底层的接口（如tf.Module、tf.layers.Layer）。**

**在对模型进行参数更新时，可以使用实例化类对象的variables和trainable_variables属性来控制参数。**

## 5. 升级TF-slim接口开发的程序

TensorFlow 2.x版本将彻底抛弃TF-slim接口，所以升级TF-slim接口程序会有较大的工作量。官方网站给出的指导建议是：如果手动将TF-slim接口程序转化为tf.layers接口实现，则可以满足基本使用；**如果想与TensorFlow 2.x版本结合得更加紧密，则可以再将其转化为tf.keras接口。**

该内容来自于《深度学习之TensorFlow工程化项目实战》一书。如果你想了解TensorFlow的更多使用技巧以及有关新旧版本的升级方法或者是更直接的实例演示，请参考此书。
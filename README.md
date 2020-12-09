# distributedTensorFlow
## 启动

- 集群维护

  首先维护 **distributeTensorFlow.py** 脚本中的服务器集群信息，可以使用本地伪集群部署：

  ```python
  ps_spec = ["localhost:2220"]
  worker_spec = ["localhost:2221"]
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
  server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
  ```

  这里是MNIST手写体识别任务，方便调试所以只维护了一台参数服务器和一个worker节点。

- 启动项目：

  维护完服务器信息后，（本地）分多个终端分别执行下列脚本：

  ```shell
  python distributedTensorflow.py --job_name="ps" --task_index=0
  python distributedTensorflow.py --job_name="worker" --task_index=0
  python distributedTensorflow.py --job_name="worker" --task_index=1
  python distributedTensorflow.py --job_name="worker" --task_index=2
  ```

## tensorflow 分布式

​	在分布式 TensorFlow 中，参与分布式系统的所有节点或者设备被总称为一个集群（cluster），一个cluster中包含很多服务器（server），每个server去执行一项任务（task），server和task是一一对应的。所以，cluster可以看成是server的集合，也可以看成是task的集合。TensorFlow 为各个task又增加了一个抽象层，将一系列相似的task集合称为一个job，比如在PS架构中，习惯称parameter server的task集合为ps，而称执行梯度计算的task集合为worker。所以cluster又可以看成是job的集合，不过这只是逻辑上的意义，具体还要看这个server真正干什么。在 TensorFlow 中，job用name（字符串）标识，而task用index（整数索引）标识，那么cluster中的每个task可以用job的name加上task的index来唯一标识。

### 分布式并行策略

​	两种主要的实现方法来分发深度学习模型训练：模型并行性或数据并行性。有时，一种方法会导致更好的应用程序性能，而有时两种方法的组合会提高性能。

![Intro Distributed Deep Learning | Xiandong](https://xiandong79.github.io/downloads/ddl1.png)

#### 模型并行

​	在模型并行性中，模型被分为不同的部分，可以并行运行它。可以在不同节点上的相同数据上运行不同的层。这种方法可以减少worker间进行通信的需求，因为worker仅需同步共享的参数。但是实际上层与层之间的运行是存在约束的：前向运算时，后面的层需要等待前面层的输出作为输入，而在反向传播时，前面的层又要受限于后面层的计算结果。所以除非模型本身很大，一般不会采用模型并行，因为模型层与层之间存在串行逻辑。但是如果模型本身存在一些可以并行的单元，那么也是可以利用模型并行来提升训练速度，模型并行性也适用于共享高速总线并具有较大模型的单个服务器中的GPU，因为每个节点的硬件约束不是一个限制。

#### 数据并行

​	在该模式下，训练集数据被划分为多个子集，并且在不同节点上的同一复制模型上运行每个子集。在该模式下，必须在批处理计算结束时同步模型参数，以确保它们正在训练一致的模型，因为预测误差是在每台机器上独立计算的。因此，每个设备都必须将所有更改的通知发送到所有其他设备的所有型号。

​	数据并行可以是同步的（synchronous），也可以是异步的（asynchronous）。所谓同步指的是所有的设备都是采用相同的模型参数来训练，等待所有设备的mini-batch训练完成后，收集它们的梯度然后取均值，然后执行模型的一次参数更新。这相当于通过聚合很多设备上的mini-batch形成一个很大的batch来训练模型，Facebook就是这样做的，但是他们发现当batch大小增加时，同时线性增加学习速率会取得不错的效果。同步训练看起来很不错，但是实际上需要各个设备的计算能力要均衡，而且要求集群的通信也要均衡，类似于木桶效应，一个拖油瓶会严重拖慢训练进度，所以同步训练方式相对来说训练速度会慢一些。异步训练中，各个设备完成一个mini-batch训练之后，不需要等待其它节点，直接去更新模型的参数，这样总体会训练速度会快很多。但是异步训练的一个很严重的问题是梯度失效问题（stale gradients），刚开始所有设备采用相同的参数来训练，但是异步情况下，某个设备完成一步训练后，可能发现模型参数其实已经被其它设备更新过了，此时这个梯度就过期了，因为现在的模型参数和训练前采用的参数是不一样的。由于梯度失效问题，异步训练虽然速度快，但是可能陷入次优解（sub-optimal training performance）。

### 分布式训练架构

​	distributed TensorFlow 包括两种架构：Parameter server architecture（就是常见的PS架构，参数服务器）和Ring-allreduce architecture。

#### PS架构

​	在Parameter server架构（PS架构）中，集群中的节点被分为两类：parameter server和worker。

![Scaling up with Distributed Tensorflow on Spark | by Benoit Descamps |  Towards Data Science](https://miro.medium.com/max/1012/1*691Sexy23zPn0Mv_T6pgBQ.png)

​	其中parameter server存放模型的参数，而worker负责计算参数的梯度。在每个迭代过程，worker从parameter sever中获得参数，然后将计算的梯度返回给parameter server，parameter server聚合从worker传回的梯度，然后更新参数，并将新的参数广播给worker。

#### **Ring-allreduce架构**

​	在Ring-allreduce架构中，各个设备都是worker，并且形成一个环，如下图所示，没有中心节点来聚合所有worker计算的梯度。在一个迭代过程，每个worker完成自己的mini-batch训练，计算出梯度，并将梯度传递给环中的下一个worker，同时它也接收从上一个worker的梯度。对于一个包含 ![[公式]](https://www.zhihu.com/equation?tex=N) 个worker的环，各个worker需要收到其它个 ![[公式]](https://www.zhihu.com/equation?tex=N-1) worker的梯度后就可以更新模型参数。

![Master-Worker Reduce (Left) and Ring AllReduce (Right). | Download  Scientific Diagram](https://www.researchgate.net/profile/Stanimire_Tomov/publication/334375961/figure/fig2/AS:779164827271168@1562778779002/Master-Worker-Reduce-Left-and-Ring-AllReduce-Right.ppm)

​	相比PS架构，Ring-allreduce架构是带宽优化的，因为集群中每个节点的带宽都被充分利用。此外，在深度学习训练过程中，计算梯度采用BP算法，其特点是后面层的梯度先被计算，而前面层的梯度慢于前面层，Ring-allreduce架构可以充分利用这个特点，在前面层梯度计算的同时进行后面层梯度的传递，从而进一步减少训练时间。

## 部署中遇到的问题及解决

- RuntimeError: Run called even after should_stop requested.

  - analysis:

    https://stackoverflow.com/questions/42960304/basic-stopatstephook-monitoredtrainingsession-usage

  - solve:

    在模型图中定义 global_step 初始化，在模型计算的时候获取step信息并跟 requested steps 对比。

- tensorflow.python.framework.errors_impl.InvalidArgumentError: From /job:worker/replica:0/task:0:
  You must feed a value for placeholder tensor 'x' with dtype float and shape [?,784]

  - analysis:

    在模型编译的时候需要对无默认值的 `placeholder` 赋值，否则无法运行模型的子图。

  - solve:

    使用 `tf.placeholder_with_default` 替换 `tf.placeholder`

- TypeError: 'sess' must be a Session;<tensorflow.python.training.monitored_session.MonitoredSession object at 0x2be710d0>

  - solve：

    https://github.com/tensorflow/tensorflow/issues/8425

- AssertionError: y_ is not in graph

  - analysis：

    通过观察发现使用 high API 会对图中的 `name` 进行包装，将其中的计算加到实际定义的 `name` 后，因此需要找寻的 name 会变成类似 `logit/BiasAdd`的形式。

  - solve:

    https://stackoverflow.com/questions/46980287/output-node-for-tensorflow-graph-created-with-tf-layers

    
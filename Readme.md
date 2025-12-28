---
debug log：
---
12.28
* hw3: ops backend implement，结构操作尽量python处理 ，算数交给底层cpp
* as_strided： 在handle句柄相同时，设置新的shape和strides， 通过修改元数据生成新的视图， 内存开销为0， 否则alloc memory
* strides在内存紧凑期间仅由shape就能确定， 句柄handle可能指向三个地方：numpy的内存， cpu'ram, gpu's 显存
* permute 仅需要np.random.shuffle(tem = range(len(stuff.shape))), 就能确定新的stuff.shape, 进而确定新的strides

12.26
* 跑起来了， 但是和checkpoint不一致
* 找到问题了，module创建顺序不同 → RNG 消费顺序不同 → 权重不同 → forward 输出不同
* 单独跑MLPResNet， 使用不同的数据预先方式：
    * 不处理： 

      <img src="images/image-10.png" alt="alt text" width="400">

      hidden-norm 从100到150， 表现力更强， 前期收敛更快， 然而结果差别不大

      <img src="images/image12.png" alt="alt text" width="400">

    * RandomFlipHorizontal:
      水平翻转强迫模型学习更一般的特征

      <img src="images/image-9.png" alt="alt text" width="400">

      真打脸， 还不如不处理，可能是epoch太少了
    * RandomCrop:

      <img src="images/image-11.png" alt="alt text" width="400">
      
      也差一些， 这个还能理解一些， 裁剪了一些特征

---

##### 一些零散的先放在这里：

这个仓库的commit records不是很干净， 因为需要不断的上传，方便colab克隆 

12.15
* ops文件ndarray, Tensor 的转换，造成困难
* detach(), 避免计算图过大，造成内存泄漏
* hw1 all pass , not totally understood
* __call__, __init__, init__, hhhhhh
* 实现nn.Linear, 对__add__ 重载 和 broadcast 手动实现有了更深的理解
---
12.16
* keepdims, self.axis, .reshape(shape).broadcast_to(XX.shape)  
    give you an exp :
<img src="images/image-2.png" alt="alt text" width="200">

* 注意传参： axis是从0开始数的 ， 注意区分和shape的区别

<img src="images/image-3.png" alt="alt text" width="500">

12.17
* 完善nn_basic.py，kaimingNorm, kaimingUniform
* Linear module finished
* batchnorm, layernorm , dropout class finished
* hw1构建的基础功能，调用他们作为原子操作，op s--> module,往下递归微分
* 在写SGD.step() 时， 对params的理解：
<img src="images/image-4.png" alt="alt text" width="500">

* grad 也是Tensor, 这是出于计算方便的工程考量，并不作为node加入计算图，实际上， 为了节约内存，常用懒汉式在BP期才加载
* 实现momentum 和 Adam, Adam's bias correction用来避免momentum策略导致的训练初期步长太小
* horizontal flip and random crop, 提高泛化能力
* dataset 读了转成ndarray, dataloader 负责iter and next
---
12.19

* 在实现ResidualBlock的过程中， nn.Sequential串联起module, Module 的封装与数据是分离的
<img src="images/image-6.png" alt="alt text" width="400">

* 一些魔法方法和自动调用的对应：
<img src="images/12.png" alt="alt text" width="350">

* 常用内置函数
<img src="images/13.png" alt = "alt text" width = "350">


* datas = self.dataset[batch_indices], dataset 传回来的是tuple，所以这里batch_indices数组传入后， 得到的datas是数组tuple
* Tensor(data) for data in datas 也是对于tuple中的一个元素操作








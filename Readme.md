debug log：
---
1.13
* 移植的时候出现问题：
![alt text](images/14.png)
  为了计算图的延续， 必须在BACKEND添加max， 支持反向传播
* add TensorOp Stack, Max, Tanh, 其中Stack最不好写， Stack.gradient还没有想明白
关键代码：
```py
 new_shape = list(args[0].shape).insert(self.axis, len(args))
 empty_in_new_shape = array_api.empty(new_shape, dtype=args[0].dtype, device = args[0].device)
 for i, arg in enumerate(args):
     slices = [slice(None)] * len(new_shape)  #slices is a data form of "[x, y, z, ...]"
                                              # slice(None) means [:]
     slices[self.axis] = i
     empty_in_new_shape[tuple(slices)] = arg 
 return empty_in_new_shape
```
1.12
* watch lecture17, update notes
---
1.6
* 开始将hw1,2,3 移植到hw4
* 代码重构的过程中， 我发现ops_logarithmic.py 部分logsumexp, 指定axes的功能实现的不是很清楚。 暂时先实现二维版本
* fun fact: max(target, axis = 2, keepdims = true), max.shape = (2, 3, 4, 5, 6), 对这种多维max是怎么生效的呢？
   * result.shape = (2, 3, 1, 5, 6), 单独看axis = 2 的位置， 有5 * 6 = 30个子元素，对于30个位置， 同时有4个平行的兄弟元素进行比较 
---
1.4 白天
* add ReduceMax, ReduceSum in cuda, hw3 finished

* 又是函数宏， 理解更深了吧, KERNEL 和 HOST 一开始声明顺序反了
* opr 需要预先创建， host_name, kernel_name, 通过函数宏直接填进去
* const CudaArray& a 里的 & 是“引用”，不是取地址, 语义上相当于 C 的 “传只读指针” (const CudaArray* a)，但用法更像别名，函数体直接写 a.foo 而不是 a->foo
* EwiseOps等函数中，CudaArray* out 输出参数习惯用指针， 主要是工程上易读性的考虑：
  * 本函数对目标的所有权： “不管new / delete， 只负责写入”
  * 提示写入
  * 可以合法位nullptr， 更容易在内部检查或者抛错
* 增添了如下调用的GPU端实现

    <img src="images/image copy 2.png" alt="alt text" width="400">

* 这是自从hw2 model， data， optimizer 分块实现以来码的最爽的一次

---

这个仓库干了这么一些事情：
---
* 通过构建计算图， 实现**自动微分**功能， 这是反向传播所依赖的基石，主要体现在hw1
* 实现经典Optimizers (Adam, Momentum...)， Regulation Method (Dropout, Corp...)， Dataset 和 DataLoader 分离， 叠叠乐的module，等
等深度学习系统基本组成部分， 主要体现在hw2
* 支持cpu，gpu端
的加速（其实是慢速，比起标准库的实现， 能够做到“没那么慢”）， 当然需要自己动手啦！ 这是hw3的
内容

commit records不是很干净， 因为需要不断的上传，方便colab克隆 

以下是在实践过程中值得深思的代码：
`Ops Summation``ops_mathematic.py`
```py
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis = self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTIOU
        input_shape = node.inputs[0].shape
        # 可以不只沿着一个维度做summation
        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        target_shape = list(out_grad.shape)
        for axis in sorted(axes):
            target_shape.insert(axis, 1)

        out_grad = out_grad.reshape(target_shape)
        # 相当于 out_grad * all_ones
        return broadcast_to(out_grad, input_shape)
```
* 无独有偶： 为什么`Transpose`,` Reshape` 做逆运算就是微分了？ 为什么`Broadcast_to`的微分恰好是`Summation`? 不是想当然的！ 因为我们考察的是每个元素在forward过程中的贡献！！ 这样就能理解为什么抽象的非数值操作也有微分的概念。

* Pytorch 提供的计算图功能， 在`autograd.py`里面被山寨， numpy提供的数据封装形式Ndarray(见 `ndarray_backend_numpy.py`), 在`ndarray.py`里面被山寨，新的`ndarray.py`下的数据， 可以用c, cuda底层语言进行比特级别的管理和优化 


1.3
* 容易混的地方： CudaArray虽然由cpu创建， 调用构造函数过后显存分配在GPU端
* cuda: EwiseSetItem, 注意grid的设置通过BASE_THREAD_NUM, 尽量发挥硬件的并行性能，　“一个线程处理一个元素”
* 启动 kernel的时候， 实参会被拷贝到GPU上的参数空间， 所以CudaVec已经拷贝过去了
* *a, *out, 就必须手动分配到GPU
---
1.2
* cuda: add naive MatMul
---
1.1
* cpu: add reduce, naive_Matmul
* debug : MatMUL_Tile, passed in a 4-dimension array, the memory is continuous only in each Tile, and i wrongly treat it as 2-d type of continuous
---
12.31. 2025 最后一天
* cpu backend support: 
  * compact() 无序存储的array 赋给紧凑存储
  * EwiseSetitem() 紧凑存储的array赋给无序, 当前实现n_dim = 0时有下溢风险
* 函数式宏， 写了一批ElementWise ops, Scalar ops, 真带劲
---
12.30
* implement getitem(), 根据切片， 返回NDArray视图, 起到虚拟分割的作用
``` python
new_strides = NDArray.compact_strides_yyj(new_shape)
new_strides = tuple(self.strides[iter] for iter in new_axes) #真闹心啊， 这两句
```
---
12.29
* imple broadcast_to with no memory cost, 从shape[i]=1扩展到n， 内存访问步长为0, :)zip is fun, it returns iterator

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








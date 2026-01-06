import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    limit = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-limit, high = limit, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, std = std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain_recommended = math.sqrt(2)
    std = gain_recommended * math.sqrt(1 / fan_in)
    return randn(fan_in, fan_out, std = std, **kwargs)
    ### END YOUR SOLUTION


'''

### 1. `fan_in` 和 `fan_out` 到底描述了什么？

`fan_in` 和 `fan_out` 描述的是神经网络中**一个层（Layer）的连接数量**。

*   **`fan_in`**：**输入连接数**。代表一个神经元接收多少个输入信号。对于一个全连接层来说，它就是前一层神经元的数量。
*   **`fan_out`**：**输出连接数**。代表一个神经元的输出会传递给多少个神经元。对于一个全连接层来说，它就是后一层神经元的数量。

#### 形象化的解释

想象一个简单的三层神经网络：

`输入层 (4个神经元)` -> `隐藏层 (8个神经元)` -> `输出层 (2个神经元)`

现在，我们把焦点放在**隐藏层**上：

*   隐藏层中的**每一个**神经元，都与输入层的**所有4个**神经元相连。所以，对于隐藏层来说，它的 `fan_in = 4`。
*   隐藏层中的**每一个**神经元的输出，都会连接到输出层的**所有2个**神经元。所以，对于隐藏层来说，它的 `fan_out = 2`。

#### 与代码的联系

在代码中，一个全连接层通常是用一个权重矩阵 `W` 来表示的。这个矩阵的形状（shape）就直接对应了 `fan_in` 和 `fan_out`。

如果隐藏层的权重矩阵 `W` 的形状是 `(fan_in, fan_out)`，那么：

*   `W.shape[0]` 就是 `fan_in` (输入维度)
*   `W.shape[1]` 就是 `fan_out` (输出维度)

所以，在你的 `xavier_uniform(fan_in, fan_out)` 函数中，`fan_in` 和 `fan_out` 就是你正在初始化的那个权重矩阵的维度。

---

### 2. 这些函数在机器学习中的作用是什么？

像 `xavier_uniform` 这样的函数，它们的作用是进行**权重初始化（Weight Initialization）**。这是训练神经网络前一个至关重要、但又经常被忽略的步骤。

#### 为什么需要“聪明”的初始化？

神经网络的训练，本质上就是通过反向传播算法不断调整权重（`W`）和偏置（`b`）。那么，在训练开始之前，这些权重应该被设置成什么值呢？

**1. 如果把权重都初始化为 0：**
*   **灾难性后果**：所有神经元在第一次前向传播时，会计算出完全相同的值。在反向传播时，它们会收到完全相同的梯度，并进行完全相同的更新。
*   **结果**：网络失去了对称性。隐藏层中的所有神经元永远都是一样的，相当于你只有一个神经元在工作，网络完全学不到东西。

**2. 如果把权重初始化为非常大的随机数：**
*   **问题**：在前向传播时，`z = Wx + b` 的值会非常大。如果你的激活函数是 `sigmoid` 或 `tanh`，它们会迅速进入**饱和区**（输出接近 -1 或 1）。
*   **结果**：在饱和区，激活函数的梯度（导数）几乎为0。这会导致**梯度消失（Vanishing Gradients）**，权重几乎无法更新，网络学习非常缓慢。

**3. 如果把权重初始化为非常小的随机数：**
*   **问题**：在前向传播时，每经过一层，信号的方差会不断缩小。信号越来越弱。
*   **结果**：在反向传播时，梯度也会逐层递减，传到前面的层时已经小到可以忽略不计。这同样会导致**梯度消失**，靠近输入层的网络层基本不学习。

#### Xavier/Glorot 和 He 初始化的诞生

为了解决以上问题，研究人员提出了“聪明”的初始化方法。其核心思想是：

> **让信息（和梯度）在网络中传播时，保持其统计特性（如方差）基本不变。既不能放大，也不能缩小。**

1.  **Xavier (或 Glorot) 初始化** (就是你代码中的 `xavier_uniform`)
    *   **提出者**：Xavier Glorot 和 Yoshua Bengio 在2010年提出。
    *   **核心思想**：它让权重的方差为 `1 / fan_in` 或者 `2 / (fan_in + fan_out)`。这样可以平衡输入和输出的方差。
    *   **适用场景**：在 `tanh` 或 `sigmoid` 这类对称的激活函数上表现非常好。
    *   **你的代码 `math.sqrt(6 / (fan_in + fan_out))`** 正是 Xavier 均匀分布初始化的标准公式，它推导出的边界值可以确保权重方差满足要求。

2.  **He 初始化**
    *   **提出者**：Kaiming He (何恺明) 等人在2015年提出。
    *   **核心思想**：它让权重的方差为 `2 / fan_in`。
    *   **适用场景**：专门为 `ReLU` 及其变种（如 Leaky ReLU）激活函数设计。因为 ReLU 函数不是对称的（它会将所有负数置为0），Xavier 初始化在 ReLU 网络中会导致方差逐层减半。He 初始化修正了这一点。

### 总结

*   `fan_in` / `fan_out` 是描述**层连接数**的参数，直接决定了**权重矩阵的形状**。
*   `xavier_uniform` 这类函数的作用是进行**权重初始化**。
*   好的权重初始化是**成功训练神经网络的基石**。它可以避免梯度消失或梯度爆炸，让网络在训练初期就处在一个“准备好学习”的良好状态，从而大大加速收敛并提高模型性能。
'''
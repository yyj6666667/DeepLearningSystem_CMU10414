from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

#already import , just directly use it
from .ops_mathematic import *


from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        res = Z - (logsumexp(Z, axes = (1,))).reshape((Z.shape[0], -1))
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        part_1 = Tensor([1]).broadcast_to(input.shape)
        softmax = exp(node) 
        grad = out_grad - softmax * (summation(out_grad, axes = (1,)).reshape((input.shape[0], 1)).broadcast_to(input.shape))
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


# 对它微分直接得到softmax哈哈， 循环了
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z) -> "NDArray":
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis = self.axes, keepdims = True)
        Z_stable = Z - Z_max.broadcast_to(Z.shape)
        log_sum_exp = array_api.log( 
                              array_api.sum( 
                                 array_api.exp(Z_stable), 
                                          axes=self.axes, 
                                          keepdims = True
                                 )                 
                        )
        result = Z_max + log_sum_exp
        return result
        
            #这里屏蔽一个原先的二维版本
            #drop max out to prevent exp explode
            # 命名时前面的Z代表这是有多组x_i 组成的矩阵
             #n_dim = Z.shape[0]
             #Z = Tensor(Z)      
             #Z_max_each_row = max(Z, axis = 1).reshape((n_dim, 1))
             #Z_row_minus_max_each_row = Z - Z_max_each_row.broadcast_to(Z.shape)
             #Z_logsumexp_each_row = log( summation(exp(Z_row_minus_max_each_row), axes=(1,)).reshape((n_dim, 1)) ) # shape: (n_dim, 1)
             #result = Z_max_each_row + Z_logsumexp_each_row
             #return result.detach()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        shape = list(Z.shape)
        axes = self.axes if self.axes is not None else tuple(range(len(shape)))
        for ax in axes:
            shape[ax] = 1
        node_new = node.reshape(shape).broadcast_to(Z.shape)
        grad = exp(Z - node_new)
        out_grad_ = out_grad.reshape(shape).broadcast_to(Z.shape)
        return out_grad_ * grad
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
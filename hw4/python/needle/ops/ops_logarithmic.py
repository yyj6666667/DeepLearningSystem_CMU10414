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
        assert isinstance(Z, NDArray), "yyj: Z must be NDArray"
        last_axis = len(Z.shape) - 1
        z_max = Z.max((last_axis,), keepdims=True)
        z_stable = Z - z_max.broadcast_to(Z.shape)
        z_sum_exp = z_stable.exp().sum(last_axis, keepdims=True)
        log_z_sum_exp = array_api.log(z_sum_exp)
        return z_stable - log_z_sum_exp.broadcast_to(Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
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
        assert isinstance(Z, NDArray), "yyj: Z must be NDArray"
        print("shape of input", Z.shape)
        Z_max = array_api.max(Z, axis = self.axes, keepdims = True)
        print("shape of Z_max:", Z_max.shape)
        Z_stable = Z - Z_max.broadcast_to(Z.shape)
        print("shape of Z_stable",  Z_stable.shape)
        log_sum_exp = array_api.log( 
                              array_api.sum( 
                                 array_api.exp(Z_stable), 
                                          axis=self.axes, 
                                          keepdims = False
                                 )                 
                        )
        print("log_sum_exp", log_sum_exp.shape)
        result = Z_max.reshape(log_sum_exp.shape) + log_sum_exp
        print("result'shape :", result.shape)
        return result
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
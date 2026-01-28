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
        last_axis = len(Z.shape) - 1
        z_max = Z.max(axis=(last_axis,), keepdims=True)
        z_stable = Z - array_api.broadcast_to(z_max, Z.shape)
        z_sum_exp = array_api.sum(array_api.exp(z_stable), axis=(last_axis,), keepdims=True)
        log_z_sum_exp = array_api.log(z_sum_exp)
        return z_stable - array_api.broadcast_to(log_z_sum_exp, Z.shape)
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


class Softmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        last_axis = len(Z.shape) - 1
        z_max = Z.max(axis=(last_axis,), keepdims=True)
        z_stable = Z - array_api.broadcast_to(z_max, Z.shape)
        z_exp = array_api.exp(z_stable)
        z_sum_exp = array_api.sum(z_exp, axis=(last_axis,), keepdims=True)
        return z_exp / array_api.broadcast_to(z_sum_exp, Z.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        s = node
        return s * (out_grad - summation(out_grad * s, axes=(len(s.shape) - 1,)).reshape(
            tuple(s.shape[:-1]) + (1,)).broadcast_to(s.shape))


def softmax(a: Tensor) -> Tensor:
    return Softmax()(a)


# 对它微分直接得到softmax哈哈， 循环了
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z) -> "NDArray":
        ### BEGIN YOUR SOLUTION
        Z_max = Z.max(axis=self.axes, keepdims=True)
        Z_stable = Z - array_api.broadcast_to(Z_max, Z.shape)
        log_sum_exp = array_api.log(
            array_api.sum(
                array_api.exp(Z_stable),
                axis=self.axes,
                keepdims=False
            )
        )
        return array_api.reshape(Z_max, log_sum_exp.shape) + log_sum_exp
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
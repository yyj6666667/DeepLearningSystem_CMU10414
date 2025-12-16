from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from ..autograd import array_api
from needle import ops

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        res = Z - (ops.logsumexp(Tensor(Z), axes = (1,))).numpy().reshape((Z.shape[0], -1))
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        part_1 = Tensor([1]).broadcast_to(input.shape)
        softmax = ops.exp(node) 
        grad = out_grad - softmax * (ops.summation(out_grad, axes = (1,)).reshape((input.shape[0], 1)).broadcast_to(input.shape))
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis = self.axes, keepdims = True)
        Z_stable = Z - Z_max
        Z_exp = array_api.exp(Z_stable)
        Z_exp_sum = array_api.sum(Z_exp, axis = self.axes, keepdims = True)
        logsumexp = array_api.log(Z_exp_sum) + Z_max
        Z_max_final = array_api.max(Z, axis = self.axes, keepdims= False)
        return logsumexp.reshape(Z_max_final.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        shape = list(Z.shape)
        axes = self.axes if self.axes is not None else tuple(range(len(shape)))

        for ax in axes:
            shape[ax] = 1

        node_new = node.reshape(shape).broadcast_to(Z.shape)
        grad = ops.exp(Z - node_new)
        out_grad_ = out_grad.reshape(shape).broadcast_to(Z.shape)
        return out_grad_ * grad
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
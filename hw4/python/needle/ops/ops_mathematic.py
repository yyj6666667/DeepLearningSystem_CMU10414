"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a --> self | self contains op
        a = node.inputs[0]
        grad = self.scalar * (a ** (self.scalar - 1)) * out_grad
        return grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
      ##  if not isinstance(node.inputs[0], NDArray) or not isinstance(
      ##      node.inputs[1], NDArray
      ##  ):
      ##      raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = -out_grad * a / (b ** 2)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar 
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    #return DivScalar(scalar)(a) 日历仙人， 首先， 继承了__call__, 真离谱，DivScalar(scalar)返回op对象
    op = DivScalar(scalar)
    #由于继承了__call__, 这里直接调用op对象就行了
    return op(a)

class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            #默认交换沿着内存增长方向的粒度最细的两个维度（即最后两个维度）
            new_axes = tuple(range(len(a.shape) - 2)) + (len(a.shape) - 1, len(a.shape) - 2)
        else :
            new_axes = list(range(len(a.shape)))
            new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
        res = NDArray.permute(a, new_axes)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes = self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        old_shape = node.inputs[0].shape
        return reshape(out_grad, old_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        out_shape = out_grad.shape


        if len(input_shape) < len(out_shape):
            input_shape = (1,) * (len(out_shape) - len(input_shape)) + input_shape
        axes = []
        for i, (dim_out, dim_in) in enumerate(zip(out_shape, input_shape)):
            if dim_in == 1 and dim_out > 1:
                axes.append(i)
        grad = summation(out_grad, axes = tuple(axes))
        grad = reshape(grad, node.inputs[0].shape)
        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


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

        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        middle_shape = list(input_shape)
        for axe in axes:
            middle_shape[axe] = 1
        out_grad = out_grad.reshape(middle_shape)
        return out_grad.broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad @ transpose(rhs)
        rhs_grad = transpose(lhs) @ out_grad
        #唉，高维张量就不合适了，有一个广播导致rank扩张的问题，
        #还不会处理
        if (len(lhs_grad.shape) > len (lhs.shape)):
            #多出来的轴在前面
            iter_tem = range(len(lhs_grad.shape) - len(lhs.shape))
            to_be_abandon = tuple(iter_tem) 
            lhs_grad = summation(lhs_grad, to_be_abandon)
        if (len(rhs_grad.shape) > len (rhs.shape)):
            #多出来的轴在前面
            iter_tem = range(len(rhs_grad.shape) - len(rhs.shape))
            to_be_abandon = tuple(iter_tem) 
            rhs_grad = summation(rhs_grad, to_be_abandon)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = out_grad / a
        return grad
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a > 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_data = node.inputs[0].realize_cached_data()
        return (Tensor(input_data > 0)) * out_grad
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)

#add Max
class Max(TensorOp):
    def __init__(self, axis = None, keepdims = False):
        self.axis = axis
        self.keepdims = keepdims

    def compute(self, a):
        return array_api.max(a, axis = self.axis, keepdims=self.keepdims)
    
    def gradient(self, out_grad, node):
        forward_res = node.realize_cached_data()
        mask_one    = node.input[0] == forward_res.reshape(-1, 1) #有点问题， 如果有多个最大值容易造成误差
        return mask_one * out_grad

def max(a, axis, keepdims):
    return Max(axis, keepdims)(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        exp_2x = array_api.exp(2 * a)
        return (exp_2x - 1) / (exp_2x + 1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_data = node.realize_cached_data()
        return out_grad * (1 - node_data ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        tem_shape = list(args[0].shape)
        new_shape = tem_shape[:self.axis] + [len(args)] + tem_shape[self.axis:]
        empty_in_new_shape = array_api.empty(new_shape, dtype=args[0].dtype, device = args[0].device)
        for i, arg in enumerate(args):
            slices = [slice(None)] * len(new_shape)  #slices is a data form of "[x, y, z, ...]"
                                                     # slice(None) means [:]
            slices[self.axis] = i
            empty_in_new_shape[tuple(slices)] = arg 
        return empty_in_new_shape
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis = self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        res_shape = []
        for i, dim in enumerate(A.shape):
            if i != self.axis:
                res_shape.append(dim)
        
        res = []
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            res.append(A[tuple(slices)].compact().reshape(res_shape)) 
            #reshape要求连续内存，（因为懒，没有写非连续版本的reshape）这里compact是安全操作
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis = self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = [slice(None)] * len(new_shape)
        for axis in self.axes :
            if axis < len(new_shape):
                new_shape[axis] = new_shape[axis] * (self.dilation + 1)
                slices[axis] = slice(None, None, self.dilation + 1)
        new_array = array_api.full(tuple(new_shape), 0, device = a._device) #创建新内存
        assert new_array.device == a.device, "a and new_array must share same device"
        new_array[tuple(slices)] = a
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            slices[axis] = slice(None, None, self.dilation + 1)
        new_array = a[tuple(slices)].compact()
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A:(N, H, W, C_in), B: (K, K, C_in, C_out), B is kernel
        # C_out 是滤波器(filters)数量，代表学习的特征数量
        # C_in  是图像的通道数， 容易理解
        assert len(A.shape) == 4 and len(B.shape) == 4, "both A and B's input dimension must be 4"
        p = self.padding
        A = A.pad(( (0, 0), (p, p), (p, p), (0, 0) )).compact()

        N, H, W, C_in = A.shape
        K ,C_out= B.shape[0], B.shape[3]
        H_new = H - K + 1
        W_new = W - K + 1
        matmul_dim = K * K * C_in

        A_col = array_api.empty((N * H_new * W_new, matmul_dim), dtype = A.dtype, device = A._device)
        for n in range(N):
            for h in range(H_new):
                for w in range(W_new):
                    A_col[(n * H_new * W_new) + h * W_new + w, :] = A[n, h : h + K, w : w + K,:].reshape((1, matmul_dim))
                                                                                                #这里会调用我写的__setItem__加速
        
        out = A_col @ B.reshape((-1, C_out))
        out = out.reshape(N, H_new, W_new, C_out)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        K    = B.shape[0]
        # compute dA
        B_flip = flip(B, (0, 1))
        B_flip = transpose(B_flip, (0, 1, 3, 2))

        dA = conv(out_grad, B_flip, stride=1, padding=K - 1 - self.padding) #为了匹配H_A, W_A 维度， 需要padding
        
        # comput dB
        A_T = transpose(A, (3, 1, 2, 0))
        out_grad_T = transpose(out_grad, (1, 2, 0, 3))
        dB_T = conv(A_T, out_grad_T, stride=1, padding=self.padding)
        dB   = transpose(dB_T, (1, 2, 0, 3))

        return dA, dB
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)




"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class  Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device = device, dtype = dtype))
        self.bias = Parameter((init.kaiming_uniform(out_features, 1, device = device, dtype = dtype)).reshape((1, out_features))) if bias else None #太抽象了哈哈哈哈
        #下次还是分开写吧， 这样写太折寿了
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias is not None:
            mul_res = ops.matmul(X, self.weight)
            reshaped_bias = self.bias.reshape((1, self.out_features))
            broadcasted_bias = ops.broadcast_to(reshaped_bias, mul_res.shape)
            return mul_res + broadcasted_bias
        else:
            return ops.matmul(X, self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X = X.reshape((X.shape[0], -1))
        return X
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            iter =  module(x)
            x = iter
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0]
        y_hot = init.one_hot(logits.shape[1], y)
        
        # Stable Cross Entropy: LogSumExp(logits) - logits_y
        lse = ops.logsumexp(logits, axes=(1,))
        logits_y = ops.summation(logits * y_hot, axes=(1,))
        
        return ops.summation(lse - logits_y) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device = device, dtype = dtype))
        self.bias   = Parameter(init.zeros(dim, device = device, dtype = dtype))
        self.running_mean = init.zeros(dim, device = device, dtype = dtype)
        self.running_var  = init.ones(dim, device = device, dtype = dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training :
            batch_mean = ops.summation(x, axes = (0,)) / x.shape[0] #(dim,)
            diff = x - batch_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            batch_var  = ops.summation(diff * diff , axes = (0,)) / x.shape[0]

            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * batch_mean).detach()
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * batch_var).detach()

            mul_by_weight = (x - batch_mean.reshape((1, self.dim)).broadcast_to(x.shape)) / ops.power_scalar(batch_var + self.eps, 0.5).reshape((1, self.dim)).broadcast_to(x.shape)

        elif self.training is False:
            mul_by_weight = (x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)) / ops.power_scalar(self.running_var + self.eps, 0.5).reshape((1, self.dim)).broadcast_to(x.shape)
        
        y = self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * mul_by_weight + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return  y

        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim #dim is number of features
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype = dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype = dtype))
        ### BEGIN YOUR SOLUTION
    
    def Varience(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes = (1,)) / x.shape[1]
        mean_b = mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        diff = x - mean_b
        sq_diff = diff * diff
        var = ops.summation(sq_diff, axes = (1,)) / x.shape[1]
        return var #(batch,)
        ### END YOUR SOLUTION

    def Std(self, var: Tensor) -> Tensor:
        std = ops.power_scalar(var + self.eps, 0.5)
        return std

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # y = w * (x - mean) / std + b
        # dis = (x - mean) / std
        mean = ops.summation(x, axes = (1,)) / x.shape[1]
        mean_b = mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        dis = (x - mean_b) / self.Std(self.Varience(x)).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        w_b = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b_b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        y = w_b * dis + b_b
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = x.shape
        bernoulli_distri = init.randb(*shape, p = 1 - self.p, device = x.device)
        b_dis = bernoulli_distri
        if self.training is True :
            x_hat = ops.mul_scalar(x * b_dis, 1 / (1 - self.p))
            return x_hat
        else :
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 #当stride = 1， kernel size为奇数时， 确保输出特征图的H， W与输入一致

        ### BEGIN YOUR SOLUTION
        #init weight_tensor using kaiming uniform
        self.fan_in = in_channels * kernel_size ** 2
        self.fan_out = out_channels * kernel_size ** 2
        self.w_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(init.kaiming_uniform(self.fan_in, self.fan_out, self.w_shape, nonlinearity="relu", device = device, dtype = dtype, requires_grad = True))
        if bias:
            bound = 1.0 / (self.in_channels * (self.kernel_size)**2) ** 0.5
            self.bias = Parameter(init.rand(out_channels, low=-bound, high=bound, requires_grad=True, device=device, dtype=dtype))
        else:
            self.bias = None # if not added in forward there will be "no attribute" error
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x:  (N, C, H, W) -> (N, H, W, C)
        x = ops.permute(x, (0, 2, 3, 1))
        res_1 = ops.conv(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            bias = ops.reshape(self.bias, (1, 1, 1, -1))
            res_1 = res_1 + bias.broadcast_to(res_1.shape)
        
        return res_1.permute((0, 3, 1, 2))
        ### END YOUR SOLUTION
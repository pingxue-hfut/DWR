import torch.nn as nn
import torch.nn.functional as F
import torch
from . import binactivation
from torch.nn import init


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.binary_activation = binactivation.BinaryActivation()

    def forward(self, input):
        w = self.weight
        a = input
        binary_weights_no_grad = torch.sign(w)
        cliped_weights = torch.clamp(w, -1.0, 1.0)
        bw = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        ba = self.binary_activation(a)
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output







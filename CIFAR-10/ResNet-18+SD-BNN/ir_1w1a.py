import torch.nn as nn
import torch.nn.functional as F
import binactivation
import torch


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.binary_activation = binactivation.BinaryActivation()
        self.alpha = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.dybeta = SELayer(in_channels)

    def forward(self, input):
        w = self.weight
        a = input
        mw = 0.01 * w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1).detach()
        w = w + torch.sigmoid(self.alpha.view(self.alpha.size(0), 1, 1, 1)) * mw
        a = a + self.dybeta(a)
        binary_weights_no_grad = torch.sign(w)
        cliped_weights = torch.clamp(w, -1.0, 1.0)
        bw = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        ba = self.binary_activation(a)

        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

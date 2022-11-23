import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import ir_1w1a
from modules import recalibration_modules


__all__ = ['resnet20_1w1a', 'resnet56_1w1a']


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class Convalg(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super(Convalg, self).__init__()
        self.convalg = nn.Sequential(
            ir_1w1a.IRConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.Hardtanh()
        )

    def forward(self, x):
        y = self.convalg(x)
        return y


class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()
        self.midchannel = planes
        self.p1 = (self.midchannel - in_planes) // 2
        self.p2 = self.midchannel - planes
        self.s10 = LearnableBias(in_planes)
        self.conv1 = ir_1w1a.IRConv2d(in_planes, self.midchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.midchannel)
        if stride != 1:
            self.shortcut1 = LambdaLayer(lambda x:
                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.p1, self.p1), "constant", 0))
        else:
            self.shortcut1 = LambdaLayer(lambda x:
                                         F.pad(x[:, :, :, :], (0, 0, 0, 0, self.p1, self.p1), "constant", 0))
        self.s11 = LearnableBias(self.midchannel)
        self.nonlinear1 = nn.PReLU(self.midchannel)
        self.s12 = LearnableBias(self.midchannel)

        self.s20 = LearnableBias(self.midchannel)
        self.conv2 = ir_1w1a.IRConv2d(self.midchannel, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut2 = LambdaLayer(lambda x:
                                     F.pad(x[:, self.p2:, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        # self.shortcut = nn.Sequential()
        self.s21 = LearnableBias(planes)
        self.nonlinear2 = nn.PReLU(planes)
        self.s22 = LearnableBias(planes)

    def forward(self, x):
        out = self.s10(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out += self.shortcut1(x)
        out = self.s11(out)
        out = self.nonlinear1(out)
        out = self.s12(out)
        x1 = self.shortcut2(out)
        out = self.s20(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x1
        out = self.s21(out)
        out = self.nonlinear2(out)
        out = self.s22(out)
        return out


class BasicBlock_1w1a2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w1a2, self).__init__()
        self.conv1 = ir_1w1a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w1a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.nonlinear = nn.Hardtanh()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ir_1w1a.IRConv2d(in_planes, self.expansion * planes, kernel_size=3, padding=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out += self.shortcut(x)
        out = self.nonlinear(out)
        x1 = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += x1
        out = self.nonlinear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.nonlinear = nn.Hardtanh()
        # self.alg = Convalg(16, 24)
        self.layer1 = self._make_layer(block, 30, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 86, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 154, num_blocks[2], stride=2)
        self.alg2 = Convalg(154, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlinear(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.alg2(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out


def resnet20_1w1a():
    # return ResNet(BasicBlock_1w1a, [3, 3, 3])
    return ResNet(BasicBlock_1w1a, [1, 1, 1])


def resnet56_1w1a():
    return ResNet(BasicBlock_1w1a, [9, 9, 9])


def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

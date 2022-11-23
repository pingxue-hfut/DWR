import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

__all__ = ['birealnet18', 'birealnet34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Convalg(nn.Module):
    def __init__(self, inplanes, planes):
        self.oup = planes
        super(Convalg, self).__init__()
        self.alg = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.prelu = nn.PReLU(planes)

    def forward(self, x):
        out = self.alg(x)
        out = torch.cat([out, x], dim=1)[:, :self.oup, :, :]
        out = self.prelu(out)
        return out


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        self.binary_activation = BinaryActivation()

    def forward(self, x):
        # real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        bx = self.binary_activation(x)
        y = F.conv2d(bx, real_weights, stride=self.stride, padding=self.padding)

        return y


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()

        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.binary_conv(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.move0 = LearnableBias(inplanes)
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 116
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.alg = Convalg(64, 116)
        self.layer1 = self._make_layer(block, 116, layers[0])
        self.layer2 = self._make_layer(block, 188, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 408, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 910, layers[3], stride=2)
        # self.adp = nn.AdaptiveAvgPool2d(7)
        # self.fusion = GhostModule(1024, 512, ratio=20)
        '''
        self.fusion = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.fbn = nn.BatchNorm2d(512)
        self.fprelu = nn.PReLU(512)
        '''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = nn.BatchNorm1d(910)
        self.fc = nn.Linear(910 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.alg(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.fc(x)
        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model


'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch
import math
from modules import ir_1w1a


__all__ = ['vgg_small_1w1a']
channel = [128, 256, 230, 512, 472]


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


class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear = nn.Hardtanh()

        self.conv1 = ir_1w1a.IRConv2d(128, channel[0], kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(channel[0])

        self.conv2 = ir_1w1a.IRConv2d(channel[0], channel[1], kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel[1])

        self.conv3 = ir_1w1a.IRConv2d(channel[1], channel[2], kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel[2])

        self.conv4 = ir_1w1a.IRConv2d(channel[2], channel[3], kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channel[3])

        self.conv5 = ir_1w1a.IRConv2d(channel[3], channel[4], kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(channel[4])

        # self.avgpool = nn.AdaptiveAvgPool2d(4)
        # self.fusion = GhostModule(640, 512, ratio=32)
        self.alg2 = Convalg(channel[4], 512)

        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)   # 128
        x = self.bn0(x)
        x = self.nonlinear(x)
        # fx = self.avgpool(x)

        x = self.conv1(x)   # 128
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)

        x = self.conv2(x)  # 256
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)  # 256
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)  # 512
        x = self.bn4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)  # 512
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        
        x = self.alg2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model

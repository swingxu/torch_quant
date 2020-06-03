import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from .. import layers as L

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def active(quantize=False, a_bit=8, maxval=3):
    if quantize:
        return L.activation_quantize_fn(a_bit=a_bit,maxval=maxval)
    else:
        return nn.ReLU(inplace=True)

def gen_conv(conv, s=1):
    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight) * s
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    conv.forward = forward
    return conv

class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        #init.kaiming_normal(m.weight)
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def _GroupNorm(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes,
            planes,conv=nn.Conv2d,bn=nn.BatchNorm2d,active=F.relu,stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     bn(self.expansion * planes)
                )
        self.active = active

    def forward(self, x):
        out = self.active(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.active(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, bn=nn.BatchNorm2d,
            conv=nn.Conv2d,num_classes=10,active=nn.ReLU(inplace=True)):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.active = active
        self.conv1 = conv(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = bn(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0],
                stride=1, conv=conv, bn=bn,active=active)
        self.layer2 = self._make_layer(block, 32, num_blocks[1],
                stride=2, conv=conv, bn=bn,active=active)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,
                conv=conv, bn=bn,active=active)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes,
            num_blocks,stride,conv=nn.Conv2d,bn=nn.BatchNorm2d,active=F.relu):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                conv=conv,bn=bn,active=active,stride=stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.active(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(WS=False, GN=False, a_quant=True, a_bit=16, a_max=3, s=1):
    if WS:
        conv = gen_conv(WSConv2d, s=s)
    else :
        conv = nn.Conv2d
    if GN:
        bn = _GroupNorm
    else:
        bn = nn.BatchNorm2d
    F_active = active(a_quant, a_bit=a_bit, maxval=a_max)
    return ResNet(BasicBlock, [3, 3, 3], conv=conv, bn=bn, active=F_active)

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


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


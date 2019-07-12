'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
from .. import layers as L
import torch
__all__ = ['alexnet']

def active(quantize=False, a_bit=8, maxval=3):
    if quantize:
        return L.activation_quantize_fn(a_bit=a_bit,maxval=maxval)
    else:
        return nn.ReLU(inplace=True)

class AlexNet(nn.Module):

    def __init__(self, num_classes=10, a_quant=False, a_bit=None, a_maxval=None):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            active(a_quant,a_bit=a_bit,maxval=a_maxval),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            active(a_quant,a_bit=a_bit,maxval=a_maxval),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            active(a_quant,a_bit=a_bit,maxval=a_maxval),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            active(a_quant,a_bit=a_bit,maxval=a_maxval),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            active(a_quant,a_bit=a_bit,maxval=a_maxval),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        print('alexnet, a_quant=%d, a_bit=%d, a_maxval=%d' %(a_quant,a_bit,a_maxval))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model

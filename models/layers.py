import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, WeightS = True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.WeightS = WeightS

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        if self.WeightS:
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                    keepdim=True).mean(dim=3, keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
            weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def uniform_quantize(k,maxval):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        sf = n/maxval
        out = torch.round(input * sf) / sf
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply

class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit, maxval):
    super(activation_quantize_fn, self).__init__()
    #assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.maxval = maxval
    self.uniform_q = uniform_quantize(k=a_bit,maxval=maxval)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, self.maxval))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


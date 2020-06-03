import torch
import models.imagenet as customized_models

# flod the weight norm layer as normal weight layer
fname = '/home/yelab/pytorch-gpu/dev/torch_quant/checkpoints/cifar10/resnet/WS_S005_A4_acc9055'
checkpoint = torch.load(fname+'.pth.tar')
for k in checkpoint['state_dict'].keys():
    if 'conv' in k or 'downsample.0' in k:
        print(k)
        weight = checkpoint['state_dict'][k]
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight) * 0.05
        checkpoint['state_dict'][k]= weight
        
torch.save(checkpoint,fname+'_wn.pth.tar')
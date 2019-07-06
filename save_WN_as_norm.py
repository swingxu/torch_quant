import torch
import models.imagenet as customized_models

# flod the weight norm layer as normal weight layer
checkpoint = torch.load('./R-50-GN-WS.pth.tar')
for k in checkpoint.keys():
    if 'conv' in k or 'downsample.0' in k:
        print(k)
        weight = checkpoint[k]
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        checkpoint[k] = weight
        
torch.save(checkpoint,'./checkpoint2.pth.tar')
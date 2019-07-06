# Implentation of Uniform/Non-Uniform Quantization of Weight Normalization based model

Under Construction.

## Usage
### Test resnet
、、、
python main_imgnet_training.py ~/workspace/dataset/torch_imagenet/CLS-LOC/ -a resnet --pretrained  --resume ./R-50-GN-WS.pth.tar -e --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
、、、

### Uniform Quantize the weight
、、、
python ./weightquant.py
、、、

### Save the Standard Weight based model as normal model
、、、
python ./save_WN_as_norm.py
、、、

## Citaiton:
* [Weight Standardlization](https://arxiv.org/abs/1903.10520)
* [AlexNet](https://arxiv.org/abs/1404.5997)
* [VGG](https://arxiv.org/abs/1409.1556) (Imported from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar))
* [ResNet](https://arxiv.org/abs/1512.03385)
* [Pre-act-ResNet](https://arxiv.org/abs/1603.05027)
* [ResNeXt](https://arxiv.org/abs/1611.05431) (Imported from [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch))
* [Wide Residual Networks](http://arxiv.org/abs/1605.07146) (Imported from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch))
* [DenseNet](https://arxiv.org/abs/1608.06993)
* [ResNeXt](https://arxiv.org/abs/1611.05431)

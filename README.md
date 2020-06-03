# Implentation of Uniform/Non-Uniform Quantization of Weight Normalization based model

Still Under Construction.

## Todo

- [x] kmeans quantization
- [x] finetuning the quantized model for cifar
- [ ] freeze the weights and training the scale factor after quantization
- [ ] finetuning the quantized model for imagenet
- [ ] expansion and compression of the weights
- [ ] depth-wise convolution quantization

## Usage

### Test resnet

```python
python main_imgnet_training.py ~/workspace/dataset/torch_imagenet/CLS-LOC/ -a resnet --pretrained  --resume ./R-50-GN-WS.pth.tar -e --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0、
```

### Train Activation quantized model for cifar10
```python
python train_cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --lr 0.01 --a_quant --a_bit 8 --checkpoint checkpoints/cifar10/alexnet_act8   
```

### Test with cifar10
```python
 python train_cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --lr 0.01 --checkpoint checkpoints/cifar10/alexnet_dorefa --resume checkpoints/cifar10/alexnet/model_best.pth.tar
```
### Save the Standard Weight based model as normal model

```
python save_WN_as_norm.py
```

### Quantize the weight

kmeans quantization：

```python
python weightquant.py --input_ckpt ./R-50-GN-WS-flod.pth.tar --quant_method kmeans --output_ckpt ./R-50-GN-WS-q --weight_bits 4 --n_sample 10000
```

linear quantization：

```python
python weightquant.py --input_ckpt ./R-50-GN-WS-flod.pth.tar --quant_method linear --output_ckpt ./R-50-GN-WS-q --weight_bits 4
```



## Citaiton:

\* [Weight Standardlization](https://arxiv.org/abs/1903.10520)

\* [AlexNet](https://arxiv.org/abs/1404.5997)

\* [VGG](https://arxiv.org/abs/1409.1556) (Imported from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar))

\* [ResNet](https://arxiv.org/abs/1512.03385)

\* [Pre-act-ResNet](https://arxiv.org/abs/1603.05027)

\* [ResNeXt](https://arxiv.org/abs/1611.05431) (Imported from [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch))

\* [Wide Residual Networks](http://arxiv.org/abs/1605.07146) (Imported from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch))

\* [DenseNet](https://arxiv.org/abs/1608.06993)

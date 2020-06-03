s=(1 0.5 0.05 0.005)
wb=4
for index in {0..1}
do
#python train_cifar.py -a resnet20 --epochs 200 --schedule 60 120 --gamma 0.1 --lr 0.1  --checkpoint checkpoints/cifar10/resnet  --weight_standard --log_name A4W32_S${s[index]} --a_quant --a_bit 4 --ws_scale ${s[index]}
  python train_cifar.py -a resnet20 --epochs 300 --w_quant --w_bit ${wb} --schedule 210 240 270 --gamma 0.1 --lr 0.1 --resume checkpoints/cifar10/resnet/A4W32_S${s[index]}.pth.tar  --checkpoint checkpoints/cifar10/resnet  --weight_standard --log_name A4W${wb}_S${s[index]}_km --a_quant --a_bit 4 --ws_scale ${s[index]}
   # train_cifar.py -a resnet20 --epochs 200 --schedule 60 120 --gamma 0.1 --lr 0.1 --checkpoint checkpoints/cifar10/resnet  --weight_standard --log_name A32W32_S${s[index]} --ws_scale ${s[index]}

done
"""
kmeans-quantization:
python weightquant.py --input_ckpt ./R-50-GN-WS-flod.pth.tar --quant_method kmeans --output_ckpt ./R-50-GN-WS-q --weight_bits 4 --n_sample 10000

linear-quantization:
python weightquant.py --input_ckpt ./R-50-GN-WS-flod.pth.tar --quant_method linear --output_ckpt ./R-50-GN-WS-q --weight_bits 4
"""
import argparse
import utils.quant as quant
import torch
from collections import OrderedDict
import numpy as np
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='PyTorch Weight Quantization')
parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh|kmeans')
parser.add_argument('--input_ckpt', default='./checkpoint2.pth.tar', help='input ckpt')
parser.add_argument('--output_ckpt', default='./quant', help='output ckpt')
parser.add_argument('--weight_bits', type=int, default=8, help='bit-width for weight')
parser.add_argument('--bn_bits', type=int, default=32, help='bit-width for running mean and std')
parser.add_argument('--n_sample', type=int, default=10000, help='trainset size to train the quantizer of the weight')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')

args = parser.parse_args()

def weight_quant(state_dict,bits=8,bn_bits=32,overflow_rate=0.01,quant_method='linear'):
    state_dict_quant = OrderedDict()
    sf_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'running' in k:
            if bn_bits >=32:
                print("Ignoring {}".format(k))
                state_dict_quant[k] = v
                continue
            else:
                bits = bn_bits
        if 'bn' in k:
            state_dict_quant[k] = v
            continue
        if quant_method == 'linear':
            sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=overflow_rate)
            if 'conv' in k:
                print(k,quant.compute_integral_part(v, overflow_rate=overflow_rate))
            v_quant  = quant.linear_quantize(v, sf, bits=bits)
        elif quant_method == 'log':
            v_quant = quant.log_minmax_quantize(v, bits=bits)
        elif quant_method == 'minmax':
            v_quant = quant.min_max_quantize(v, bits=bits)
        else:
            v_quant = quant.tanh_quantize(v, bits=bits)
        state_dict_quant[k] = v_quant
    #torch.save(state_dict_quant,'./checkpoint2_q.pth.tar')
    return state_dict_quant

class kmeansQuant():
    def __init__(self, bits=8, trainset_size=10000):
        self.trainset_size = trainset_size
        self.bits = bits
        train_data = np.random.normal(size=trainset_size)
        min_ = min(train_data)
        max_ = max(train_data)
        space = np.linspace(min_, max_, num=2**bits)
        self.kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        self.kmeans.fit(train_data.reshape(-1,1))
 
    def quant(self,checkpoint):
        for key,tensor in checkpoint.items():
            dev = tensor.device
            checkpoint[key] = self.kmeans.cluster_centers_[self.kmeans.predict(tensor.reshape(-1,1).cpu().numpy())]
            checkpoint[key] = torch.from_numpy(checkpoint[key].reshape(tensor.shape)).to(dev)
        return checkpoint

if __name__ == '__main__':
    state_dict = torch.load(args.input_ckpt)
    if args.quant_method == 'kmeans':
        kmq = kmeansQuant(bits=args.weight_bits,trainset_size=args.n_sample)
        state_dict = kmq.quant(state_dict)
    elif args.quant_method == 'linear':
        state_dict = weight_quant(state_dict,bits=args.weight_bits,quant_method='linear',overflow_rate=args.overflow_rate)
    torch.save(state_dict,args.output_ckpt+args.quant_method+str(args.weight_bits)+'.pth.tar')
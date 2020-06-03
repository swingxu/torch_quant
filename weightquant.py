"""
kmeans-quantization:
python weightquant.py --input_ckpt ./R-50-GN-WS-flod.pth.tar --quant_method kmeans --output_ckpt ./R-50-GN-WS-q --weight_bits 4 --n_sample 10000

linear-quantization:
python weightquant.py --input_ckpt ./R-50-GN-WS-flod.pth.tar --quant_method linear --output_ckpt ./R-50-GN-WS-q --weight_bits 4

channelwise-quantization
python weightquant.py --input_ckpt ./checkpoints/imagenet/resnet18.pth.tar --quant_method channelwise_quant --output_ckpt ./checkpoints/resnet18_channelwise_linear --weight_bits 4
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
        if 'conv' in k and 'weight' in k:
            sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=overflow_rate)
            v_quant  = quant.linear_quantize(v, sf, bits=bits)
            state_dict_quant[k] = v_quant
            print(k, bits)
        else:
            if 'running' in k or 'bn' in k or 'num_batches_tracked' in k or 'fc' in k:
                print("Ignoring {}".format(k))
            state_dict_quant[k] = v
            continue
    #torch.save(state_dict_quant,'./checkpoint2_q.pth.tar')
    return state_dict_quant

def nw_weight_quant(state_dict,bits=8,bn_bits=32,overflow_rate=0.01,quant_method='linear'):
    state_dict_quant = OrderedDict()
    for k, v in state_dict.items():
        if 'conv' in k and 'weight' in k:
            _v = v.flatten()
            try:
                v_quant = torch.cat((v_quant,_v), dim=0)
            except:
                v_quant = _v
        else:
            continue
    del _v
    sf = bits - 1. - quant.compute_integral_part(v_quant, overflow_rate=overflow_rate)
    for k, v in state_dict.items():
        if 'conv' in k and 'weight' in k:
            v_quant  = quant.linear_quantize(v, sf, bits=bits)
            state_dict_quant[k] = v_quant
            print(k, bits)
        else:
            if 'running' in k or 'bn' in k or 'num_batches_tracked' in k or 'fc' in k:
                print("Ignoring {}".format(k))
            state_dict_quant[k] = v
            continue
    #torch.save(state_dict_quant,'./checkpoint2_q.pth.tar')
    return state_dict_quant

def channelwise_quant(state_dict,bits=8,bn_bits=32,overflow_rate=0.01,quant_method='linear'):
    state_dict_quant = OrderedDict()
    sf_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'conv' in k and 'weight' in k:
            for vv in v:
                sf = bits - 1. - quant.compute_integral_part(vv, overflow_rate=overflow_rate)
                v_kernel_quant  = quant.linear_quantize(vv, sf, bits=bits)
                sp = v_kernel_quant.shape
                try:
                    v_quant = torch.cat((v_quant,v_kernel_quant.reshape(1,sp[0],sp[1],sp[2])), dim=0)
                except:
                    v_quant = v_kernel_quant.reshape(1,sp[0],sp[1],sp[2])
            state_dict_quant[k] = v_quant
            del v_quant
        else:
            if 'running' in k or 'bn' in k or 'num_batches_tracked' in k or 'fc' in k:
                print("Ignoring {}".format(k))
            state_dict_quant[k] = v
            continue
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
            if 'conv' in key and 'weight' in key:
                print(key)
                dev = tensor.device
                checkpoint[key] = self.kmeans.cluster_centers_[self.kmeans.predict(tensor.reshape(-1,1).cpu().numpy())]
                checkpoint[key] = torch.from_numpy(checkpoint[key].reshape(tensor.shape)).to(dev)
        return checkpoint

class kmeansQuant_t():
    def __init__(self, bits=8, trainset=None):
        self.trainset = trainset
        self.bits = bits
        min_ = trainset.min()
        max_ = trainset.max()
        space = np.linspace(min_, max_, num=2**bits)
        self.kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        self.kmeans.fit(trainset.reshape(-1,1))
        
    def quant(self, checkpoint):
        for key,tensor in checkpoint.items():
            if 'conv' in key and 'weight' in key:
                print(key)
                dev = tensor.device
                checkpoint[key] = self.kmeans.cluster_centers_[self.kmeans.predict(tensor.reshape(-1,1).cpu().numpy())]
                checkpoint[key] = torch.from_numpy(checkpoint[key].reshape(tensor.shape)).to(dev)
        return checkpoint

class kmeansQuant_tn():
    def __init__(self, bits=8, trainset=None):
        self.trainset = trainset
        self.bits = bits
        min_ = trainset.min()
        max_ = trainset.max()
        space = np.linspace(min_, max_, num=2**bits)
        self.kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        self.kmeans.fit(trainset.reshape(-1,1))
        
    def quant(self, _tensor):
        dev = _tensor.device
        t_tensor = self.kmeans.cluster_centers_[self.kmeans.predict(_tensor.reshape(-1,1).cpu().numpy())]
        _tensor = torch.from_numpy(t_tensor.reshape(_tensor.shape)).to(dev)
        return _tensor
    
if __name__ == '__main__':
    state_dict = torch.load(args.input_ckpt)
    if args.quant_method == 'kmeans':
        kmq = kmeansQuant(bits=args.weight_bits,trainset_size=args.n_sample)
        _state_dict = kmq.quant(state_dict['state_dict'])
        state_dict['state_dict']=_state_dict
    elif args.quant_method == 'kmeans_dg':
        trainset = np.array(0)
        for k, v in state_dict['state_dict'].items():
            if 'conv' in k and 'weight' in k:
                trainset = np.append(trainset,v.flatten().cpu().detach().numpy())
        print(trainset.shape)
        kmq = kmeansQuant_t(bits=args.weight_bits,trainset=trainset)
        _state_dict = kmq.quant(state_dict['state_dict'])
    elif args.quant_method == 'kmeans_dn':
        for k, v in state_dict['state_dict'].items():
            if 'conv' in k and 'weight' in k:
                kmq = kmeansQuant_tn(bits=args.weight_bits,trainset=v.flatten().cpu().detach().numpy())
                state_dict['state_dict'][k] = kmq.quant(v)
    elif args.quant_method == 'linear':
        _state_dict = weight_quant(state_dict['state_dict'],bits=args.weight_bits,quant_method='linear',overflow_rate=args.overflow_rate)
        state_dict['state_dict']=_state_dict
    elif args.quant_method == 'channelwise_quant':
        _state_dict = channelwise_quant(state_dict['state_dict'],bits=args.weight_bits,quant_method='linear',overflow_rate=args.overflow_rate)
        state_dict['state_dict']=_state_dict
    elif args.quant_method == 'nw_quant':
        _state_dict = nw_weight_quant(state_dict['state_dict'],bits=args.weight_bits,quant_method='linear',overflow_rate=args.overflow_rate)
        state_dict['state_dict']=_state_dict
    torch.save(state_dict,args.output_ckpt+args.quant_method+str(args.weight_bits)+'.pth.tar')
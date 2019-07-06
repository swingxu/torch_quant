
import argparse
import quant
import torch
from collections import OrderedDict


def weight_quant(state_dict,bits=8,bn_bits=32,overflow_rate=0.02,quant_method='linear'):
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
    torch.save(state_dict_quant,'./checkpoint2_q.pth.tar')
    return

state_dict = torch.load('../pytorch-classification/checkpoint2.pth.tar')
weight_quant(state_dict,bits=8)
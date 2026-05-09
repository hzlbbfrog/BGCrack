import math
from os import device_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

def get_freq_indices(method):

    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,1,2,3,4,5,6,7,8,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,2,3,4,5,6,7,8,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]

        #all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        #all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16', Active = 'GELU'):
        super(MultiSpectralAttentionLayer, self).__init__()
        
        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU
            
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 8) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.Max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            Active_Layer(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.fc2 = nn.Sequential(
           nn.Conv2d(channel, channel//reduction, 1, bias=False),
           Active_Layer(inplace=True),
           nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.w= nn.Parameter(torch.ones(2))
        

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x

        y = self.dct_layer(x_pooled) 
        y_max = self.Max_pool(y) 
        y_sum =  torch.sum(y, dim=[2,3], keepdim=True) 

        y_max_out = self.fc1(y_max)
        y_sum_out = self.fc2(y_sum)
        y_out = self.w[0]*y_max_out+self.w[1]*y_sum_out

        out = x_pooled * self.sigmoid(y_out)
        return out


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)

        self.height = height
        self.width = width
        self.c0 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.c1 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.c2 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.c3 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.c4 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.c5 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.c6 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.c7 = nn.Conv2d(channel, channel//8, kernel_size=1, stride=1, padding=0, bias=False)

        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        #n, c, h, w = x.shape

        r0 = self.c0(x * self.weight[0,:,:])
        r1 = self.c1(x * self.weight[1,:,:])
        r2 = self.c2(x * self.weight[2,:,:])
        r3 = self.c3(x * self.weight[3,:,:])
        r4 = self.c4(x * self.weight[4,:,:])
        r5 = self.c5(x * self.weight[5,:,:])
        r6 = self.c6(x * self.weight[6,:,:])
        r7 = self.c7(x * self.weight[7,:,:])

        rs = torch.cat([r0, r1, r2, r3, r4, r5, r6, r7], dim=1)

        return rs

    def build_filter(self, pos, freq, POS): # pos = position
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):

        dct_filter = torch.zeros(16, self.height, self.width)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter
import math
from os import device_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

def get_freq_indices(method):

    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,1,2,3,4,5,6,7,8,9,10,11]
        mapper_x = all_top_indices_x[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x

class MultiSpectralAttentionLayer_spatial(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16', Active = 'GELU'):
        super(MultiSpectralAttentionLayer_spatial, self).__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x = get_freq_indices(freq_sel_method)
        mapper_x = [temp_x * (channel // 8) for temp_x in mapper_x] 

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, channel)

        self.conv1 = nn.Conv2d(16, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        max, sum = self.dct_layer(x_pooled) 

        y_out = torch.cat([max, sum], dim=1)
        y_out = self.conv1(y_out)

        y = self.sigmoid(y_out) 
        
        out = x_pooled * y

        return out


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        #n, c, h, w = x.shape

        dct0 = x * self.weight[0,:,:,:]
        dct1 = x * self.weight[1,:,:,:]
        dct2 = x * self.weight[2,:,:,:]
        dct3 = x * self.weight[3,:,:,:]
        dct4 = x * self.weight[4,:,:,:]
        dct5 = x * self.weight[5,:,:,:]
        dct6 = x * self.weight[6,:,:,:]
        dct7 = x * self.weight[7,:,:,:]

        r0,_= torch.max(dct0, dim=1, keepdim=True)
        r1,_= torch.max(dct1, dim=1, keepdim=True) 
        r2,_= torch.max(dct2, dim=1, keepdim=True) 
        r3,_= torch.max(dct3, dim=1, keepdim=True) 
        r4,_= torch.max(dct4, dim=1, keepdim=True) 
        r5,_= torch.max(dct5, dim=1, keepdim=True) 
        r6,_= torch.max(dct6, dim=1, keepdim=True) 
        r7,_= torch.max(dct7, dim=1, keepdim=True) 
        result_max = torch.cat([r0, r1, r2, r3, r4, r5, r6, r7], dim=1)

        r0_s= torch.sum(dct0, dim=1, keepdim=True)
        r1_s= torch.sum(dct1, dim=1, keepdim=True) 
        r2_s= torch.sum(dct2, dim=1, keepdim=True)
        r3_s= torch.sum(dct3, dim=1, keepdim=True) 
        r4_s= torch.sum(dct4, dim=1, keepdim=True)
        r5_s= torch.sum(dct5, dim=1, keepdim=True) 
        r6_s= torch.sum(dct6, dim=1, keepdim=True)
        r7_s= torch.sum(dct7, dim=1, keepdim=True)
        result_sum = torch.cat([r0_s, r1_s, r2_s, r3_s, r4_s, r5_s, r6_s, r7_s], dim=1)

        return result_max, result_sum

    def build_filter(self, pos, freq, POS): # pos = position
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, channel):

        dct_filter = torch.zeros(8, channel, tile_size_x,tile_size_y)
        for i in range(8):
            for t_x in range(channel):
                u_x = mapper_x[i]
                dct_filter[i, t_x, : ,:] = self.build_filter(t_x, u_x, channel)
                        
        return dct_filter
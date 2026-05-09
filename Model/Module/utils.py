import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class Map_2_Grad(nn.Module):
    def __init__(self):
        super(Map_2_Grad, self).__init__()

        self.fx, self.fy = self.get_fx_fy()

    def get_fx_fy(self): 
        fx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]).astype(np.float32) 
        fy = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]).astype(np.float32)
        fx = np.reshape(fx, (1, 1, 3, 3))
        fy = np.reshape(fy, (1, 1, 3, 3))
        fx = Variable(torch.from_numpy(fx))
        fy = Variable(torch.from_numpy(fy))
        fx = fx.cuda() 
        fy = fy.cuda()
        return fx, fy
    
    def forward(self, p):
        
        pred = p.sigmoid()
        pred = F.pad(pred, (1, 1, 1, 1), mode='replicate') 
        
        pred_fx = F.conv2d(pred, self.fx)
        pred_fy = F.conv2d(pred, self.fy)
        num = pred_fx*pred_fx + pred_fy*pred_fy + 1e-6
        pred_grad = (num).sqrt()
        
        return pred_grad


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

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LayerNorm_He(nn.Module):
    """  
    Inputs with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean((1,2,3), keepdim=True)
        s = (x - u).pow(2).mean((1,2,3), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNorm_d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

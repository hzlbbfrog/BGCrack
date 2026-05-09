import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Label_2_Grad(nn.Module):
    def __init__(self, device):
        super(Label_2_Grad, self).__init__()
        
        self.device = device
        self.fx, self.fy = self.get_fx_fy()

    def get_fx_fy(self):                
        fx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]).astype(np.float32) 
        fy = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]).astype(np.float32)
        fx = np.reshape(fx, (1, 1, 3, 3))
        fy = np.reshape(fy, (1, 1, 3, 3))
        fx = Variable(torch.from_numpy(fx))
        fy = Variable(torch.from_numpy(fy))
        fx = fx.to(self.device) 
        fy = fy.to(self.device)
        return fx, fy
    
    def forward(self, label):
        
        label = label.gt(0.5).float() # find the bigger location
        label = F.pad(label, (1, 1, 1, 1), mode='replicate') # padding. 保证卷积完尺度不变。 # print(pred.shape) torch.Size([3, 1, 514, 514])
        label_fx = F.conv2d(label, self.fx)
        label_fy = F.conv2d(label, self.fy)
        label_grad = torch.sqrt(label_fx*label_fx + label_fy*label_fy + 1e-6)
        return label_grad



    
    








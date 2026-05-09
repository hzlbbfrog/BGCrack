import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Module.Spatial_maxsum import MultiSpectralAttentionLayer_spatial
from .Module.Channel_maxsum import MultiSpectralAttentionLayer
from .Module.MobileViT import MobileViTBlock
from .Module.utils import Map_2_Grad

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class GELU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        
    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.reduce(x)


class Smaller3(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, groups=channel, bias=False)
        )
        self.conv1 = nn.Sequential(
            BasicConv2d(in_channel+channel, channel, 1),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x0, x_s):
        x_s = self.down4(x_s)
        x = self.conv1(torch.cat((x0,x_s),1))
        return x

class Bigger1(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(channel, channel, 2, stride=2)
        )
        self.conv1 = nn.Sequential(
            BasicConv2d(in_channel+channel, channel, 1),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
    def forward(self, x0,x_e):
        x_e = self.up4(x_e)
        x = self.conv1(torch.cat((x0,x_e),1))
        return x


class conv_upsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class CrossOptimizationModule(nn.Module):
    # Cross Refinement Unit
    def __init__(self, channel, Active):
        super().__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU

        self.conv1 = conv_upsample(channel)
        self.conv2 = conv_upsample(channel)
        self.conv3 = conv_upsample(channel)
        self.conv4 = conv_upsample(channel)
        self.conv5 = conv_upsample(channel)
        self.conv6 = conv_upsample(channel)
        self.conv7 = conv_upsample(channel)
        self.conv8 = conv_upsample(channel)
        self.conv9 = conv_upsample(channel)
        self.conv10 = conv_upsample(channel)
        self.conv11 = conv_upsample(channel)
        self.conv12 = conv_upsample(channel)

        self.conv_f1_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f1_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f1_3 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f1_4 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f2_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f2_3 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f2_4 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f3_3 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f3_4 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_f4_4 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )

        self.conv_ef1_1 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef1_2 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef1_3 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef1_4 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef2_2 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef2_3 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef2_4 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef3_3 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef3_4 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_ef4_4 = nn.Sequential(
            BasicConv2d(channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.w_f1 = nn.Parameter(torch.ones(5))
        self.w_f2 = nn.Parameter(torch.ones(4))
        self.w_f3 = nn.Parameter(torch.ones(3))
        self.w_f4 = nn.Parameter(torch.ones(2))
        self.w_e1 = nn.Parameter(torch.ones(5))
        self.w_e2 = nn.Parameter(torch.ones(4))
        self.w_e3 = nn.Parameter(torch.ones(3))
        self.w_e4 = nn.Parameter(torch.ones(2))
        
    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):

        x_sf1_1 = self.conv_f1_1(torch.cat((x_s1,x_e1),1))
        x_sf1_2 = self.conv_f1_2(torch.cat((x_s1,self.conv1(x_e2, x_s1)),1))
        x_sf1_3 = self.conv_f1_3(torch.cat((x_s1,self.conv2(x_e3, x_s1)),1))
        x_sf1_4 = self.conv_f1_4(torch.cat((x_s1,self.conv3(x_e4, x_s1)),1))
        x_sf1 = self.w_f1[0]*x_s1+self.w_f1[1]*x_sf1_1+self.w_f1[2]*x_sf1_2+self.w_f1[3]*x_sf1_3+self.w_f1[4]*x_sf1_4
        #x_sf1 = self.conv_f1(torch.cat((x_s1, x_sf1_1, x_sf1_2, x_sf1_3, x_sf1_4), 1))

        x_sf2_2 = self.conv_f2_2(torch.cat((x_s2,x_e2),1))
        x_sf2_3 = self.conv_f2_3(torch.cat((x_s2,self.conv4(x_e3, x_s2)),1))
        x_sf2_4 = self.conv_f2_4(torch.cat((x_s2,self.conv5(x_e4, x_s2)),1))
        x_sf2 = self.w_f2[0]*x_s2+self.w_f2[1]*x_sf2_2+self.w_f2[2]*x_sf2_3+self.w_f2[3]*x_sf2_4
        #x_sf2 =  self.conv_f2(torch.cat((x_s2, x_sf2_2, x_sf2_3, x_sf2_4), 1))

        x_sf3_3 = self.conv_f3_3(torch.cat((x_s3,x_e3),1))
        x_sf3_4 = self.conv_f3_4(torch.cat((x_s3,self.conv6(x_e4, x_s3)),1))
        x_sf3 = self.w_f3[0]*x_s3+self.w_f3[1]*x_sf3_3+self.w_f3[2]*x_sf3_4
        #x_sf3 = self.conv_f3(torch.cat((x_s3, x_sf3_3, x_sf3_4), 1))

        x_sf4_4 = self.conv_f4_4(torch.cat((x_s4,x_e4),1))
        x_sf4 = self.w_f4[0]*x_s4+self.w_f4[1]*x_sf4_4
        #x_sf4 = self.conv_f4(torch.cat((x_s4, x_sf4_4), 1))

        x_ef1_1 = self.conv_ef1_1(x_e1 * x_s1)
        x_ef1_2 = self.conv_ef1_2(x_e1 * self.conv7(x_s2, x_e1))
        x_ef1_3 = self.conv_ef1_3(x_e1 * self.conv8(x_s3, x_e1))
        x_ef1_4 = self.conv_ef1_4(x_e1 * self.conv9(x_s4, x_e1))
        x_ef1 = self.w_e1[0]*x_e1+self.w_e1[1]*x_ef1_1+self.w_e1[2]*x_ef1_2+self.w_e1[3]*x_ef1_3+self.w_e1[4]*x_ef1_4
        #x_ef1 = self.conv_f5(torch.cat((x_e1, x_ef1_1, x_ef1_2, x_ef1_3, x_ef1_4), 1))

        x_ef2_2 = self.conv_ef2_2(x_e2 * x_s2)
        x_ef2_3 = self.conv_ef2_3(x_e2 * self.conv10(x_s3, x_e2))
        x_ef2_4 = self.conv_ef2_4(x_e2 * self.conv11(x_s4, x_e2))
        x_ef2 = self.w_e2[0]*x_e2+self.w_e2[1]*x_ef2_2+self.w_e2[2]*x_ef2_3+self.w_e2[3]*x_ef2_4
        #x_ef2 = self.conv_f6(torch.cat((x_e2, x_ef2_2, x_ef2_3, x_ef2_4), 1))

        x_ef3_3 = self.conv_ef3_3(x_e3 * x_s3)
        x_ef3_4 = self.conv_ef3_4(x_e3 * self.conv12(x_s4, x_e3))
        x_ef3 = self.w_e3[0]*x_e3+self.w_e3[1]*x_ef3_3+self.w_e3[2]*x_ef3_4
        #x_ef3 = self.conv_f7(torch.cat((x_e3, x_ef3_3, x_ef3_4), 1))

        x_ef4_4 = self.conv_ef4_4(x_e4 * x_s4)
        x_ef4 = self.w_e4[0]*x_e4+self.w_e4[1]*x_ef4_4
        #x_ef4 = self.conv_f8(torch.cat((x_e4, x_ef4_4), 1))

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4


class SelfFusionModule(nn.Module):
    # Cross Refinement Unit
    def __init__(self, channel, Active):
        super().__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU
        
        self.conv_s3_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_s2_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_s1_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        
        self.conv_s2_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_s3_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_s4_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        
        self.s4_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.s3_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.s2_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.e4_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.e3_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.e2_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.s1_down2 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, groups=channel, bias=False)
        self.s2_down2 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, groups=channel, bias=False)
        self.s3_down2 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, groups=channel, bias=False)
        self.e1_down2 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, groups=channel, bias=False)
        self.e2_down2 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, groups=channel, bias=False)
        self.e3_down2 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, groups=channel, bias=False)
        
        self.conv_e3_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_e2_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_e1_1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_e2_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_e3_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        self.conv_e4_2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 1, padding=0),
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            Active_Layer(inplace=True),
        )
        
        self.SiLU1 = nn.SiLU()
        self.SiLU2 = nn.SiLU()
        
    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        
        x_s3_1 = self.conv_s3_1(torch.cat((x_s3, self.s4_up2(x_s4)),1))
        x_s2_1 = self.conv_s2_1(torch.cat((x_s2, self.s3_up2(x_s3_1)),1))
        x_s1_1 = self.conv_s1_1(torch.cat((x_s1, self.s2_up2(x_s2_1)),1))
        
        x_s2_2 = self.conv_s2_2(torch.cat((x_s2, self.s1_down2(x_s1)),1))
        x_s3_2 = self.conv_s3_2(torch.cat((x_s3, self.s2_down2(x_s2_2)),1))
        x_s4_2 = self.conv_s4_2(torch.cat((x_s4, self.s3_down2(x_s3_2)),1))
        
        
        x_e3_1 = self.conv_e3_1(torch.cat((x_e3, self.e4_up2(x_e4)),1))
        x_e2_1 = self.conv_e2_1(torch.cat((x_e2, self.e3_up2(x_e3_1)),1))
        x_e1_1 = self.conv_e1_1(torch.cat((x_e1, self.e2_up2(x_e2_1)),1))
        
        x_e2_2 = self.conv_e2_2(torch.cat((x_e2, self.e1_down2(x_e1)),1))
        x_e3_2 = self.conv_e3_2(torch.cat((x_e3, self.e2_down2(x_e2_2)),1))
        x_e4_2 = self.conv_e4_2(torch.cat((x_e4, self.e3_down2(x_e3_2)),1))

        new_x_s1 = x_s1_1+x_s1
        new_x_s2 = x_s2_1+x_s2_2
        new_x_s3 = x_s3_1+x_s3_2
        new_x_s4 = x_s4+x_s4_2
        
        new_x_e1 = x_e1_1+x_e1
        new_x_e2 = x_e2_1+x_e2_2
        new_x_e3 = x_e3_1+x_e3_2
        new_x_e4 = x_e4+x_e4_2
        
        return new_x_s1, new_x_s2, new_x_s3, new_x_s4, new_x_e1, new_x_e2, new_x_e3, new_x_e4


class FeatureFusionModule_E(nn.Module): 
    def __init__(self, channel, Active):
        super().__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU

        self.up1 = nn.ConvTranspose2d(channel, channel, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(channel, channel, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(channel, channel, 2, stride=2)

        self.up01 = nn.ConvTranspose2d(channel, channel, 2, stride=2)
        self.up02 = nn.ConvTranspose2d(channel, channel, 2, stride=2)

        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=7, stride=1, padding=3, groups=2*channel, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.Conv2d(2*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat2 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=7, stride=1, padding=3, groups=2*channel, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.Conv2d(2*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat3 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=7, stride=1, padding=3, groups=2*channel, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.Conv2d(2*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat02 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=7, stride=1, padding=3, groups=2*channel, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.Conv2d(2*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat01 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.output01 = nn.Conv2d(channel, 1, 1)

    def forward(self, x0, x1, x2, x3, x4):
        x3 = torch.cat((x3,self.up1(x4)), 1)
        x3 = self.conv_cat1(x3) 

        x2 = torch.cat((x2,self.up2(x3)), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1,self.up3(x2)), 1) 
        x1 = self.conv_cat3(x1) 

        x0 = torch.cat((x0,self.up02(x1)), 1) 
        x0 = self.conv_cat02(x0) 

        x0 = self.up01(x0) 
        x0 = self.conv_cat01(x0) 

        x = self.output01(x0) #

        return x


class FeatureFusionModule_B(nn.Module): 
    def __init__(self, channel, Active): 
        super().__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU
        
        self.up1 = nn.ConvTranspose2d(channel, channel, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(channel, channel, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(channel, channel, 2, stride=2)

        self.up01 = nn.ConvTranspose2d(channel, channel, 2, stride=2)
        self.up02 = nn.ConvTranspose2d(channel, channel, 2, stride=2)

        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(3*channel, 3*channel, kernel_size=7, stride=1, padding=3, groups=3*channel, bias=False),
            nn.BatchNorm2d(3*channel),
            nn.Conv2d(3*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat2 = nn.Sequential(
            nn.Conv2d(3*channel, 3*channel, kernel_size=7, stride=1, padding=3, groups=3*channel, bias=False),
            nn.BatchNorm2d(3*channel),
            nn.Conv2d(3*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat3 = nn.Sequential(
            nn.Conv2d(3*channel, 3*channel, kernel_size=7, stride=1, padding=3, groups=3*channel, bias=False),
            nn.BatchNorm2d(3*channel),
            nn.Conv2d(3*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat4 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=7, stride=1, padding=3, groups=2*channel, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.Conv2d(2*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat02 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=7, stride=1, padding=3, groups=2*channel, bias=False),
            nn.BatchNorm2d(2*channel),
            nn.Conv2d(2*channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.conv_cat01 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1), 
            Active_Layer(inplace=True)
        )
        self.output01 = nn.Conv2d(channel, 1, 1)


    def forward(self, x0, x1, x2, x3, x4, s1, s2, s3, s4):
        x4 = torch.cat((x4,s4), 1)
        x4 = self.conv_cat4(x4)

        x3 = torch.cat((x3,self.up3(x4),s3), 1)
        x3 = self.conv_cat3(x3) 

        x2 = torch.cat((x2,self.up2(x3),s2), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1,self.up1(x2),s1), 1) 
        x1 = self.conv_cat1(x1) 

        x0 = torch.cat((x0,self.up02(x1)), 1)  
        x0 = self.conv_cat02(x0) 

        x0 = self.up01(x0) 
        x0 = self.conv_cat01(x0) 

        x = self.output01(x0)

        return x


class Stage1(nn.Module): # Fist Block
    def __init__(self, in_ch, out_ch, Active):
        super().__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, groups=out_ch, bias=False),
            Active_Layer(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class DoubleConv(nn.Module): 
    def __init__(self, in_ch, out_ch, Active):
        super().__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU

        self.conv1= nn.Conv2d(in_ch, in_ch, kernel_size=7, stride=1, padding=3, groups=in_ch, bias=False)
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv2= nn.Sequential( # 3*3
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False), 
            Active_Layer(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class metaformer(nn.Module):
    def __init__(self, in_ch, Active): # 'ortho backward' 
        super().__init__()
        
        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU
            
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.token_mixer = nn.Conv2d(in_ch, in_ch, kernel_size=7, stride=1, padding=3, groups=in_ch, bias=False)
        self.norm2 = nn.BatchNorm2d(in_ch)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//4, 1, bias=False),
            Active_Layer(inplace=True),
            nn.Conv2d(in_ch//4, in_ch, 1, bias=False)
        )
        
    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x 
        

class ResBlock_fft_bench(nn.Module):
    def __init__(self, in_ch, Active, norm='forward'): # 'ortho backward' 
        super().__init__()
        
        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU

        self.former3 = metaformer(2*in_ch, Active)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            Active_Layer(inplace=True)
        )
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm) # y.real, N C H W/2+1, real part||   # y.imag, N C H W/2+1, imaginary part||   # default: last two dimensions
        y_f = torch.cat([y.real, y.imag], dim=1) # y_f, N C*2 H W/2+1
        y_f = self.former3(y_f) # y, N C*2 H W/2+1
        y_real, y_imag = torch.chunk(y_f, 2, dim=1) # y_real y_imag = N C H W/2+1
        y_out = torch.complex(y_real, y_imag) # N C H W/2+1
        y_out = torch.fft.irfft2(y_out, s=(H, W), norm=self.norm) # N C H W S
        y_out = self.conv4(y_out)
        return y_out


class Catfusion(nn.Module): 
    def __init__(self,channel, Active):
        super().__init__()
        
        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'SiLU':
            Active_Layer = nn.SiLU
            
        self.conv1= nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            Active_Layer(inplace=True)
        )
        self.w = nn.Parameter(torch.ones(3))
        
    def forward(self, x0, x1, x2):
        x = x0*self.w[0]+x1*self.w[1]+x2*self.w[2]
        x = self.conv1(x)

        return x


class Output_g(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1= nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1) 
        self.pre_gradient = Map_2_Grad()        
        
    def forward(self, pred_b):
        gradient = self.pre_gradient(pred_b) 
        gradient = self.conv1(gradient)
        
        return gradient

class Output_b(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1= nn.Sequential(
            BasicConv2d(2, 2, 3, padding=1),
            nn.Conv2d(2, 1, 1)
        )      
        
    def forward(self, pred_b, pred_e):
        
        x = torch.cat((pred_b,pred_e), 1)
        x = self.conv1(x)
        return x


class BGCrack(nn.Module):
    def __init__(self, channel=32):
        super().__init__()

        filter = [32, 64, 128, 256, 512] 
        # ------------- Encoder, Downsample Block -------------- #
        self.conv1 = Stage1(3, filter[0],'SiLU')  # /2
        self.pool1 = nn.MaxPool2d(2) # /4
        self.conv2 = DoubleConv(filter[0], filter[1],'SiLU') 
        self.pool2 = nn.MaxPool2d(2) # /8
        self.conv3 = DoubleConv(filter[1], filter[2],'SiLU') 
        self.pool3 = nn.MaxPool2d(2) # /16
        self.conv4 = DoubleConv(filter[2], filter[3],'SiLU') 
        self.pool4 = nn.MaxPool2d(2) # /32
        self.conv5 = DoubleConv(filter[3], filter[4],'SiLU') 
        
        self.HFIE1_S = MultiSpectralAttentionLayer_spatial(64, 128, 128,  reduction=16, freq_sel_method = 'top16', Active = 'SiLU')
        self.HFIE2_S = MultiSpectralAttentionLayer_spatial(128, 64, 64,  reduction=16, freq_sel_method = 'top16', Active = 'SiLU')
        self.HFIE1_C = MultiSpectralAttentionLayer(64, 128, 128,  reduction=8, freq_sel_method = 'top16', Active = 'SiLU')
        self.HFIE2_C = MultiSpectralAttentionLayer(128, 64, 64,  reduction=8, freq_sel_method = 'top16',Active = 'SiLU')
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))

        self.edge1 = Reduction(filter[1], channel)
        self.edge2 = Reduction(filter[2], channel)
        self.edge3 = Smaller3(filter[3], channel)
        self.edge4 = Smaller3(filter[4], channel)

        self.body1 = Bigger1(filter[1], channel)
        self.body2 = Bigger1(filter[2], channel)
        self.body3 = Reduction(filter[3], channel)
        self.body4 = Reduction(filter[4], channel)

        self.MVT3 = MobileViTBlock(dim=48, depth=1, channel=32, channel_out=32, kernel_size=3, patch_size=(2, 2), mlp_dim=48*2)
        self.b3_FFT = ResBlock_fft_bench(channel,'SiLU') 
        self.catfusion3 = Catfusion(channel,'SiLU') 
        self.MVT4 = MobileViTBlock(dim=48, depth=1, channel=32, channel_out=32, kernel_size=3, patch_size=(2, 2), mlp_dim=48*2)
        self.b4_FFT = ResBlock_fft_bench(channel,'SiLU') 
        self.catfusion4 = Catfusion(channel,'SiLU') 

        self.SFM = SelfFusionModule(channel,'SiLU')
        self.COM = CrossOptimizationModule(channel,'SiLU') 

        self.FFM_b = FeatureFusionModule_B(channel,'SiLU') 
        self.FFM_e = FeatureFusionModule_E(channel,'SiLU')

        self.output_g = Output_g()
        self.output_b = Output_b()

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.conv1(x) # size 1/2
        x1 = self.pool1(x0) # size 1/4
        x1 = self.conv2(x1) # size 1/4 

        x2 = self.pool2(x1) 
        x2 = self.conv3(x2) # size 1/8 

        x3 = self.pool3(x2) 
        x3 = self.conv4(x3) # size 1/16
        
        x4 = self.pool4(x3)
        x4 = self.conv5(x4) # size 1/32

        # -------------------------- Edge -------------------------- #
        x1_HFIE_s = self.HFIE1_S(x1)
        x1_HFIE_c = self.HFIE1_C(x1)
        x1_att = x1_HFIE_s * self.w1[0] + x1_HFIE_c * self.w1[1] # or concat
        
        x2_HFIE_s = self.HFIE2_S(x2)
        x2_HFIE_c = self.HFIE2_C(x2)
        x2_att = x2_HFIE_s * self.w2[0] + x2_HFIE_c * self.w2[1]
        
        # 降维
        x_e1 = self.edge1(x1_att) # 256→32
        x_e2 = self.edge2(x2_att) # 512→32
        x_e3 = self.edge3(x3, x_e1) # →32
        x_e4 = self.edge4(x4, x_e2) # →32

        # -------------------------- Body -------------------------- #
        # 降维
        x_b3_0 = self.body3(x3) # 256→32
        x_b4_0 = self.body4(x4) # 512→32

        x_b3_1 = self.b3_FFT(x_b3_0)
        x_b3_2 = self.MVT3(x_b3_0)
        x_b3 = self.catfusion3(x_b3_0, x_b3_1, x_b3_2)
        
        x_b4_1 = self.b4_FFT(x_b4_0)
        x_b4_2 = self.MVT4(x_b4_0)
        x_b4 = self.catfusion4(x_b4_0, x_b4_1, x_b4_2)

        x_b1 = self.body1(x1, x_b3) # 64→32
        x_b2 = self.body2(x2, x_b4) # 128→32

        # four cross refinement units
        x1_b1, x1_b2, x1_b3, x1_b4, x1_e1, x1_e2, x1_e3, x1_e4 = self.SFM(x_b1, x_b2, x_b3, x_b4, x_e1, x_e2, x_e3, x_e4)
        x1_b1_out = x1_b1 + x_b1
        x1_b2_out = x1_b2 + x_b2
        x1_b3_out = x1_b3 + x_b3
        x1_b4_out = x1_b4 + x_b4
        x1_e1_out = x1_e1 + x_e1
        x1_e2_out = x1_e2 + x_e2
        x1_e3_out = x1_e3 + x_e3
        x1_e4_out = x1_e4 + x_e4

        x2_b1, x2_b2, x2_b3, x2_b4, x2_e1, x2_e2, x2_e3, x2_e4 = self.COM(x1_b1_out, x1_b2_out, x1_b3_out, x1_b4_out, x1_e1_out, x1_e2_out, x1_e3_out, x1_e4_out)
        x2_b1_out = x2_b1 + x1_b1_out
        x2_b2_out = x2_b2 + x1_b2_out
        x2_b3_out = x2_b3 + x1_b3_out
        x2_b4_out = x2_b4 + x1_b4_out
        x2_e1_out = x2_e1 + x1_e1_out
        x2_e2_out = x2_e2 + x1_e2_out
        x2_e3_out = x2_e3 + x1_e3_out
        x2_e4_out = x2_e4 + x1_e4_out
        
        # feature aggregation using u-net
        pred_b = self.FFM_b(x0, x2_b1_out, x2_b2_out, x2_b3_out, x2_b4_out, x2_e1_out, x2_e2_out, x2_e3_out, x2_e4_out)
        pred_e = self.FFM_e(x0, x2_e1_out, x2_e2_out, x2_e3_out, x2_e4_out)

        pred_b_out = self.output_b(pred_b, pred_e)
        
        pred_g = self.output_g(pred_b_out)
        
        return pred_b_out.sigmoid(), pred_e.sigmoid(), pred_g

# Other packages
import numpy as np
import os
import argparse
import cv2
from datetime import datetime
import copy
import time

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data

# user defined
from Dataset.Steel_cracks import get_Steel_cracks, Get_Steel_Cracks_with_Edge
from utils.metrics import get_IoU, get_Dice

from ptflops import get_model_complexity_info


def Get_Options():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0) # 添加一个local_rank参数
    parser.add_argument('--dataset', default='steel_cracks_with_edge', help='steel_cracks_with_edge/steel_cracks/Deepcrack')
    parser.add_argument('--modelname', default='BGCrack', help='model names')
    parser.add_argument('--epoch', type=int, default=70, help='epoch number')
    parser.add_argument('--lr', type=float, default=6e-3, help='learning rate') 
    parser.add_argument('--batchsize', type=int, default=9, help='batch size')
    parser.add_argument('--save_path', default='./Checkpoints/')  #
    parser.add_argument("--log_dir", default='Result_log', help="log dir") 
    parser.add_argument("--test_epoch", type=int, default=66)
    parser.add_argument("--times", type=str, default='2')
    opt = parser.parse_args()
    
    return opt


def Get_Dataset(opt):

    Current_dir = os.getcwd().replace('\\','/')
    
    if opt.dataset =='steel_cracks':
        Dataset = get_Steel_cracks
    elif opt.dataset =='steel_cracks_with_edge':
        Dataset = Get_Steel_Cracks_with_Edge
    
    test_dataset = Dataset(Current_dir,'Test')
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False) # 对于Val和Test不用分布式
    
    return test_loader


def Get_Model(opt):

    if opt.modelname == 'BGCrack':
        from Model.BGCrack import BGCrack

    model = BGCrack().to(device)
    
    return model


if __name__ =="__main__":
    
    opt = Get_Options()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Get Dataset
    test_loader = Get_Dataset(opt)
    
    # Get Model
    model = Get_Model(opt)
    
    # Load parameters
    save_path = opt.save_path + opt.dataset + opt.modelname + '/' + opt.times +'/'
    load_name = save_path + 'lr=' + str(opt.lr) + '_batchsize=' + str(opt.batchsize)+ '_epoch=' + str(opt.test_epoch) + '.pth'
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(load_name, map_location=device).items()}, strict=True)
    model.eval()
    
    # Define savad path
    save_path = './Tr/' + opt.modelname +'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    with torch.no_grad(): 
        
        Dice_total = 0
        IoU_total = 0
        num = len(test_loader) #验证集图片的总数
        print(f'number of test images is {num}')  
        
        for i, pack in enumerate(test_loader, start=1): 
            
            image, gt, name = pack 
            image = Variable(image).to(device)
            gt = gt.to(device)
            

            predict, pre_edge, edge = model(image)
            
            predict = predict.squeeze() 
            predict_save = predict.data.cpu().numpy() 
            
            # ------------- Save the prediction -------------- #
            cv2.imwrite(os.path.join(save_path, name[0]+'.png'), predict_save*255.0)
            
            # ------------- Calculate Dice -------------- #
            gt = gt.squeeze() # gt: numpy, (448, 448)
            Dice = get_Dice(gt, predict)
            Dice_total += Dice
            
            # ------------- Calculate IoU -------------- #
            IoU = get_IoU(gt, predict)
            IoU_total += IoU
            
        # ------------- Print the IoU and Dice -------------- #
        print('mi IoU=%f' % (IoU_total / num))
        print('mi Dice=%f' % (Dice_total / num))
        print(f"Dice coefficient = {Dice_total / num}")
        
        # ------------- Count the parameters -------------- #
        total = sum([param.nelement() for param in model.parameters()]) 
        print("Number of parameter: %.2fM" % (total/1e6))

        # ------------- Calculate the computational complexity -------------- #
        model1 = Get_Model(opt)
        # ------------- https://github.com/sovrasov/flops-counter.pytorch ------------- #
        macs, params = get_model_complexity_info(model1, (3, 512, 512), as_strings = False, print_per_layer_stat = False, verbose = False)
        print("# ------------- ptflops -------------- #")
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print(f"params = {params/1e6}M")
        print(f"macs = {macs/1e9}G")
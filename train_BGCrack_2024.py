# Other packages
import numpy as np
import os
import cv2
import argparse
import logging
from datetime import datetime
from ptflops import get_model_complexity_info

# torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data

# user defined
from Dataset.Steel_cracks import get_Steel_cracks, Get_Steel_Cracks_with_Edge
from utils.func import Label_2_Grad
from utils.loss import dice_loss_function, CharbonnierLoss
from utils.metrics import get_Dice


def Get_Options():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0) # 添加一个local_rank参数
    parser.add_argument('--dataset', default='steel_cracks_with_edge', help='steel_cracks_with_edge/steel_cracks/Deepcrack')
    parser.add_argument('--modelname', default='BGCrack', help='model names')
    parser.add_argument('--epoch', type=int, default=70, help='epoch number')
    parser.add_argument('--times', type=str, default='1')
    parser.add_argument('--lr', type=float, default=6e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=9, help='batch size')
    parser.add_argument('--save_path', default='./Checkpoints/') 
    parser.add_argument('--image_save_path', default='./Train_image/') 
    parser.add_argument("--log_dir", default='Result_log', help="log dir")
    opt = parser.parse_args()
    
    return opt


def Get_Log(opt):
    dirname = os.path.join(opt.log_dir, opt.dataset, opt.modelname)
    filename = dirname + '/Modelname=' + opt.modelname + ' Bacthsize=' + str(opt.batchsize) + ' TotalEpoch=' + str(opt.epoch) + '.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level = logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging


def Get_Dataset(opt):

    Current_dir = os.getcwd().replace('\\','/')
    
    if opt.dataset =='steel_cracks':
        Dataset = get_Steel_cracks
    elif opt.dataset =='steel_cracks_with_edge':
        Dataset = Get_Steel_Cracks_with_Edge
        
    train_dataset = Dataset(Current_dir,'Train')
    train_sampler = data.distributed.DistributedSampler(train_dataset,num_replicas=num_gpus,rank=opt.local_rank) # 切分训练数据集
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt.batchsize//num_gpus, shuffle=False, sampler=train_sampler)
    
    val_dataset = Dataset(Current_dir,'Validation')
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False) # 对于Val和Test不用分布式
    
    return train_loader, train_sampler, val_loader


def Get_Model(opt):

    if opt.modelname == 'BGCrack':
        from Model.BGCrack import BGCrack

    model = BGCrack().to(device)
    
    return model
    

def Validation(opt, model, val_loader, best_epoch, epoch, best_dice):
    
    with torch.no_grad(): 
        model= model.eval()
        
        i=0   #验证集中第i张图
        Dice_total = 0
        num = len(val_loader) #验证集图片的总数
        print(f'number of val images is {num}')  
        logger.info(f'number of val images is {num}')
        
        if best_epoch == 0:
            # ------------- Calculate macs and params. https://github.com/sovrasov/flops-counter.pytorch ------------- #
            macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings = False, print_per_layer_stat = False, verbose = False)
            print("# ------------- ptflops -------------- #")
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            print(f"macs = {macs/1e9}G")
            print(f"params = {params/1e6}M")
            logging.info("# ------------- ptflops -------------- #")
            logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
            logging.info(f"macs = {macs/1e9}G")
            logging.info(f"params = {params/1e6}M")
            
        
        for i, pack in enumerate(val_loader, start=1): 
            
            image, gt, name = pack 
            image = Variable(image).to(device)
            gt = gt.to(device)

            
            predict, pred_edge, edge = model(image)
                
            predict = predict.data.squeeze() 
            
            # ------------- Calculate Dice -------------- #
            gt = gt.squeeze() 
            Dice = get_Dice(gt, predict)
            Dice_total += Dice
        
        aver_dice = Dice_total / num
        
        print(f'Current Epoch = {epoch}, aver_dice = {aver_dice:f}')
        logger.info(f'Current Epoch = {epoch}, aver_dice = {aver_dice:f}')
        
        if (epoch > 60):
            torch.save(model.state_dict(), save_path + 'lr=' + str(opt.lr) + '_batchsize=' + str(opt.batchsize)+ '_epoch=' + str(epoch) + '.pth')
        
        if aver_dice > best_dice:
            
            print(f"aver_dice={aver_dice:f} > best_dice={best_dice:f}\n")
            logger.info(f"aver_iou={aver_dice:f} > best_dice={best_dice:f}")
            
            best_dice = aver_dice
            best_epoch = epoch
            
            # saving models
            torch.save(model.state_dict(), save_path + 'lr=' + str(opt.lr) + '_batchsize=' + str(opt.batchsize)+ '_epoch=' + str(epoch) + '.pth')
            
            print('===========>save best model!')
            logging.info('===========>save best model!')
        
        print(f"Best_dice = {best_dice:f}, Best_Epoch = {best_epoch}\n")
        logger.info(f"Best_dice = {best_dice:f}, Best_Epoch = {best_epoch}\n")
        
        return best_epoch, best_dice, aver_dice


if __name__ =="__main__":
    # ------------- No.0 Train ------------- #
    opt = Get_Options()
    logger = Get_Log(opt)
    
    print('=============== Start training ===============')
    print(f'dataset = {opt.dataset}, model = {opt.modelname}, total epoch = {opt.epoch},batch size = {opt.batchsize}, Learning rate = {opt.lr}\n')    
    logger.info(f'dataset = {opt.dataset}, model = {opt.modelname}, total epoch = {opt.epoch}, batch size = {opt.batchsize}, Learning rate = {opt.lr}\n')
    print('==============================================')
    print('=' * 30)
    
    save_path = opt.save_path + opt.dataset + opt.modelname + '/' + opt.times +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    image_save_path = opt.image_save_path + opt.modelname
    if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

    # ------------- No.1 分布式 ------------- #
    # 1. 从外面得到local_rank参数，在调用DDP的时候，其会根据调用gpu自动给出这个参数
    local_rank = opt.local_rank
    num_gpus = torch.cuda.device_count()
    # 2. 定义设备
    device = torch.device('cuda:%d' % local_rank)
    print(device)
    # 3. Set device
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',world_size=num_gpus, rank=local_rank)

    # ------------- No.2 Get dataset ------------- #
    train_loader, train_sampler, val_loader = Get_Dataset(opt)

    # ------------- No.3 Build model ------------- #
    model = Get_Model(opt)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device) 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank, ],output_device=local_rank)

    # ------------- No.4 Define loss and optimizer ------------- #
    CE1 = torch.nn.BCELoss()
    CE2 = torch.nn.BCELoss()
    Charbonnier_loss = CharbonnierLoss()
    Label2Grad = Label_2_Grad(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # ------------- training ------------- #
    loss1, loss2, loss3, dice_loss1, dice_loss2, dice_loss3 = 0, 0, 0, 0, 0, 0
    best_epoch, best_dice, aver_dice = 0, 0, 0

    total_step = len(train_loader) # 训练集单GPU的训练batch总数
    print(f'total_step = {total_step}')
    print(len(train_loader.dataset)) # 训练集中图片的总数
    
    for epoch in range(opt.epoch):
        Current_epoch = epoch + 1
        
        train_sampler.set_epoch(epoch) # 这个很关键！做到真正的数据shuffle
        model = model.train()
        
        for i, pack in enumerate(train_loader, start=1): 
            optimizer.zero_grad() 

            images, gts, gt_edges, name = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device) 
            gt_edges = Variable(gt_edges).to(device)
            gt_grads = Label2Grad(gts)# edge prediction
            
            pred_sal, pred_edge, grad = model(images)
            
            loss1 = CE1(pred_sal, gts)
            loss2 = CE2(pred_edge, gt_edges)
            
            loss3 = Charbonnier_loss(grad, gt_grads)
            dice_loss1 = dice_loss_function(pred_sal, gts)
            dice_loss2 = dice_loss_function(pred_edge, gt_edges)
                
            loss = loss1 + loss2 + loss3 + dice_loss1 + dice_loss2
                
            loss.backward()
            optimizer.step() 

            if i % 100 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, dice_loss1: {:.4f}, Loss2: {:.4f}, dice_loss2: {:.4f}, Loss3: {:.4f}, dice_loss3: {:.4f},'.
                    format(datetime.now(), Current_epoch, opt.epoch, i, total_step, loss1, dice_loss1, loss2, dice_loss2, loss3, dice_loss3))
                logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, dice_loss1: {:.4f}, Loss2: {:.4f}, dice_loss2: {:.4f}, Loss3: {:.4f}, dice_loss3: {:.4f},'.
                    format(datetime.now(), Current_epoch, opt.epoch, i, total_step, loss1, dice_loss1, loss2, dice_loss2, loss3, dice_loss3))

        if Current_epoch >= opt.epoch * 0.6:
            best_epoch, best_dice, aver_dice = Validation(opt, model, val_loader, best_epoch, Current_epoch, best_dice)
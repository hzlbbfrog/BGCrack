import torch
import numpy as np

def get_Dice(gt, predict):

    #gt = np.array(gt).astype('float')# Convert Image to numpy array, print(gt.shape) #(352,352)

    intersection = torch.sum(predict*gt)
    epsilon = 1e-6
    Dice = (2.*intersection + epsilon) / (torch.sum(predict) + torch.sum(gt) + epsilon)

    return Dice


def get_IoU(image_mask, predict):
    
    predict = predict.ge(0.5).float()
    
    image_mask = image_mask.ge(0.5).float()

    epsilon = 1e-6
    interArea = predict*image_mask
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = torch.sum(interArea)
    union = torch.sum(unionArea)
    IoU = (inter + epsilon) / (union + epsilon)

    return IoU


def get_IoU_numpy(image_mask, predict):
    
    #image_mask = np.array(image_mask).astype('float')/255.0 # Convert Image to numpy array, [0,255]→[0,1] # print(gt.shape) #(352,352)
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 0~1
                predict[row, col] = 0.
            else:
                predict[row, col] = 1.

    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 0.5:   #0~255
                image_mask[row, col] = 0.
            else:
                image_mask[row, col] = 1.

    epsilon = 1e-6
    interArea = predict*image_mask
    # interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    IoU = (inter + epsilon) / (union + epsilon)
    
    return IoU

import os, glob, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class get_Steel_cracks(data.Dataset):
    def __init__(self, Current_dir, state):
        
        data_path = Current_dir + '/Dataset/' + 'Steel_cracks/' + state 
        image_root = data_path + '/images/'
        gt_root = data_path + '/masks/'
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        #print(self.images )
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),   # [0,255] -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # [0,1] -> [-1,1]

        self.gt_transform = transforms.Compose([transforms.ToTensor()]) # [0,255] -> [0.,1.]

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        name = self.images[index].split('/')[-1][0:-4]
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class Get_Steel_Cracks_with_Edge(data.Dataset):
    def __init__(self, Current_dir, state):
        self.state = state
        
        data_path = Current_dir + '/Dataset/' + 'Steelcrack/' + state 
        image_root = data_path + '/images/'
        gt_root = data_path + '/masks/'
        gt_edge_root = data_path + '/edges/'
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.gt_edges = [gt_edge_root + f for f in os.listdir(gt_edge_root) if f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gt_edges = sorted(self.gt_edges)
        
        self.size = len(self.images)
        #print(self.images )
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),   # [0,255] -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # [0,1] -> [-1,1]

        self.gt_transform = transforms.Compose([transforms.ToTensor()]) # [0,255] -> [0.,1.]

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        gt_edge = self.binary_loader(self.gt_edges[index])
        
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        gt_edge = self.gt_transform(gt_edge)
        
        name = self.images[index].split('/')[-1][0:-4]
        if self.state == 'Train':
            return image, gt, gt_edge, name
        else:
            return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size









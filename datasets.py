#import glob
#import random
#import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root_base, root_txt, root_img, root_gallery,
                 img_size   =None, 
                 mode       ="train"):
        self.img_size     = img_size
        self.mode         = mode
        self.root_base    = root_base
        self.root_gallery = root_gallery
        self.root_img     = root_img
        self.img_mode     = 'RGB'
        self.transform = transforms.Compose(
                [transforms.Resize((self.img_size, self.img_size), Image.BICUBIC),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ] ) 
        txtFiles = open(root_base + '/' + root_txt , 'r')
        files = []
        if self.mode == "train":
            for line in txtFiles:
                line = line.rstrip()
                words = line.split()

                files.append(('contaminate//img_align_rand//'+words[0], int(words[1])))
                #files.append(('contaminate//img_align_mask//MASK_'+words[0], int(words[1])))
                #files.append(('contaminate//img_align_mask_sv//MASK_SV_'+words[0], int(words[1])))
                #files.append(('contaminate//img_align_sun_glass//SG_'+words[0], int(words[1])))  
            self.files = files 
            np.random.shuffle(self.files) #random.shuffle(self.files) 
            self.files =self.files[:len(self.files)//5]
                               
        else:
            for line in txtFiles:
                line = line.rstrip()
                words = line.split()
                files.append(('contaminate//img_align_rand//'+words[0], int(words[1]))) 
                #files.append(('contaminate//img_align_mask//MASK_'+words[0], int(words[1])))
                #files.append(('contaminate//img_align_mask_sv//MASK_SV_'+words[0], int(words[1])))
                #files.append(('contaminate//img_align_sun_glass//SG_'+words[0], int(words[1])))    

            self.files = files 
            np.random.shuffle(self.files) 
            self.files =self.files[:500]
           
    def __getitem__(self, index):
            
        fn, label = self.files[index % len(self.files)]
        root_fn = self.root_base + '/' + self.root_img + '/' + fn        
        masked_img = Image.open(root_fn)        
        fn = self.root_base + '//' + self.root_img + '//complete//' + fn[-10:]
        img = Image.open(fn)
        if (img.mode!=self.img_mode):
            img = img.convert("RGB")
            masked_img = masked_img.convert("RGB")
        
        if self.mode == "train": 
            degree = np.random.randint(-15, 15)
            masked_img = masked_img.rotate(degree)
            img = img.rotate(degree)
        
        img = self.transform(img)   
        masked_img = self.transform(masked_img) 
        return masked_img, img

    def __len__(self):
        return len(self.files)

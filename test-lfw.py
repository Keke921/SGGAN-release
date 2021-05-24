# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:52:06 2020
@author: Li Mengke
"""
import argparse
import os
import numpy as np
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from model.VAE import *

import torch
import pytorch_ssim
import pandas as pd
from math import log10  

def set_seed(seed):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def read_list(list_path):
    img_list = []
    label_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
            label_list.append(img_path[1])
    print('There are {} images..'.format(len(img_list)))
    return img_list, label_list

def save_img(save_path, img_name, img):
    img_path = os.path.join(save_path, img_name)
    img_dir  = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    #fname = fname + '.png'
    save_image(img, "%s.png" %(fname), normalize=True)    
  
def test(img_list,model1,model2):
    
    Alignment = ImgAlignment()    
    count     = 0    
    outimg_save_path = './output/%s/img'%opt.test_set
    os.makedirs(outimg_save_path, exist_ok=True)
    os.makedirs('./output/%s/img_occ'%opt.test_set, exist_ok=True)
    os.makedirs('./output/%s/img_gt'%opt.test_set, exist_ok=True)
    os.makedirs('output/%s/statistics'%opt.test_set, exist_ok=True)
    
    results = {'RANDOM': {'psnr': [], 'ssim': []}, 'occ_img': {'psnr': [], 'ssim': []}}
    np.random.shuffle(img_list)
    
    test_time = []
    for img_name in img_list:
        count = count + 1
        
        img_fn = opt.dataset + 'lfw_sun_glass//'+ img_name[:-3]+'png'  #can change to other kind of occlusions by choosing other folder 
        img = Image.open(img_fn)
        fn_gt  = opt.dataset +'//lfw//' +img_name[:-3]+'png' 
        img_gt = Image.open(fn_gt)
        
        '''
        try:
            img = Alignment(img) 
        except Exception:
            print("not align")
        else:
            img_gt = Alignment(img_gt)   
        '''
        img = transform(img)
        img_gt = transform(img_gt)  
              
        if opt.cuda:
            img = img.cuda()
            img_gt = img_gt.cuda()
        with torch.no_grad():              
            img = Variable(img).unsqueeze(0) 
            img_gt = Variable(img_gt).unsqueeze(0) 
            start = time.time()
            img_tilde, _, _  = model1(img)
            img_out = model2(img_tilde,img)
            end = time.time() - start
        test_time.append(end)
        save_img(outimg_save_path, img_name, img_out)  
        save_img('./output/%s/img_occ'%opt.test_set, img_name, img)
        save_img('./output/%s/img_gt'%opt.test_set, img_name, img_gt)
        
        #计算量化指标
        #生成图片
        mse_img = ((img_gt - img_out) ** 2).data.mean()
        psnr_img = 10 * log10(1 / mse_img)
        ssim_img = pytorch_ssim.ssim(img_out, img_gt).item()
        #原始图片
        mse_img_occ = ((img_gt - img) ** 2).data.mean()
        psnr_img_occ = 10 * log10(1 / mse_img_occ)
        ssim_img_occ = pytorch_ssim.ssim(img, img_gt).item()
        
        # save psnr\ssim
        results['RANDOM']['psnr'].append(psnr_img)
        results['RANDOM']['ssim'].append(ssim_img)
        results['occ_img']['psnr'].append(psnr_img_occ)
        results['occ_img']['ssim'].append(ssim_img_occ)

        if count%1000==0:
            print("{}({}/{}).".format(os.path.join(opt.dataset, img_name), count, len(img_list), end))
        
        if count==5000:
            break
    time_total = np.array(test_time)
    f = open("output/%s/statistics/time.txt"%opt.test_set, "a")  
    print("[total time %f] [mean time %f] [count: %d]" % (time_total.sum(), time_total.mean(), count), file = f ) 
    f.close() 
    sta_save_path = 'output/%s/statistics/test_results_allvalues.npy'%opt.test_set
    np.save(sta_save_path,results)

    saved_results = {'psnr-mean': [], 'ssim-mean': [],'psnr-max': [], 'ssim-max': []}
    for item in results.values(): 
        psnr = np.array(item['psnr'])
        ssim = np.array(item['ssim'])
        if (len(psnr) == 0) or (len(ssim) == 0):
            psnr_mean = 'No data'
            ssim_mean = 'No data'
            psnr_max = 'No data'
            ssim_max = 'No data'
        else:
            psnr_mean = psnr.mean()
            ssim_mean = ssim.mean()
            psnr_max = psnr.max()
            ssim_max = ssim.max()
        saved_results['psnr-mean'].append(psnr_mean)
        saved_results['ssim-mean'].append(ssim_mean)
        saved_results['psnr-max'].append(psnr_max)
        saved_results['ssim-max'].append(ssim_max)
    out_path = 'output/%s/statistics/'%opt.test_set
    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'test_results.csv', index_label='DataSet')      
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="E:\data\prepared_image\processed_lfw_aligned", help="name of the dataset")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--img_list', default='E:/data/identity_lfw.txt', type=str, metavar='PATH', help='list of face images for feature extraction (default: none).')
    parser.add_argument('--cuda', '-c', default=True)
    parser.add_argument('--test_set', default='lfw2')
    parser.add_argument('--stage1_resume', default="./output/gen1.pth")
    parser.add_argument('--stage2_resume', default="./output/gen2.pth")
    opt = parser.parse_args(args=[])
    #print(opt)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")

    # Initialize generator and discriminator
    DIM = 256
    Z_DIM = 512 
    vae = VAE(opt.channels, DIM, Z_DIM)
    gen = Gen(opt.channels, DIM)

    # load model checkpoint
    vae.load_state_dict(torch.load(opt.stage1_resume))
    gen.load_state_dict(torch.load(opt.stage2_resume)) 

    if cuda:
        vae.cuda()
        gen.cuda()
    vae.eval()
    gen.eval()
    
    # Configure data loader          
    transforms_ = [
        transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
        #transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    set_seed(123)
    # ----------
    #  Testing
    # ----------  
    img_list,label_list = read_list(opt.img_list)
    transform = transforms.Compose(transforms_)
    test(img_list, vae, gen)

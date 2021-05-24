# -*- coding: utf-8 -*-
import argparse
import os
import torch
from torch.autograd import Variable
#import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
import numpy as np
import math
import pandas as pd

from datasets import *
from model.VAE import VAE
import pytorch_ssim

#%%
def set_seed(seed):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
#%%  训练函数
# ----------------
#  Training VAE
# ----------------  
def train(epoch):
    vae.train()
    running_results = {'batch_sizes': 0, 'img_Loss': 0, 'z_loss': 0, 'vae_loss': 0}
    out_path = images_out_path+'/training_results/' 
    if not os.path.exists(out_path):
        os.makedirs(out_path) 
        
    for i, (mask_img, img_batch) in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i
        batch_size = img_batch.size(0)
        # Configure input   
        mask_img = Variable(mask_img.type(FloatTensor))  
        imgs = Variable(img_batch.type(FloatTensor))  
        running_results['batch_sizes'] += batch_size 
        del img_batch
        # -----------------
        #  Train Generator
        # -----------------    
        x_tilde, Lz,_ = vae(mask_img)
        Lx = Recons(x_tilde, imgs)/batch_size
        loss = Lx+Lz
        optimizer.zero_grad()        
        loss.backward()                
        optimizer.step()
        
        torch.cuda.empty_cache()
        
        # loss for current batch before optimization 
        running_results['img_Loss'] += Lx.item() * batch_size
        running_results['z_loss'] += Lz.item() * batch_size
        running_results['vae_loss'] += loss.mean().item() * batch_size
        
        #  Log Progress
        if batches_done > 0 and (batches_done % 200) == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [VAE Loss: %f] [Lz: %f]"  
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(),Lz) )
            #recon_loss[batches_done//200] = loss.item()
            f = open("output/train_loss.txt", "a")  
            print(
                "[Epoch %d/%d] [Batch %d/%d] [VAE Loss: %f] [Lz: %f]"  
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(),Lz),file = f)
            f.close() 
            
        # clear cache
        del loss#, D_loss
        torch.cuda.empty_cache()    
    img_grid = torch.cat((mask_img[:6], imgs[:6], x_tilde[:6]), 0) 
    img_grid = (img_grid.cpu().data + 1) / 2       
    save_image(img_grid, out_path+"/train_epoch_%d.png" % epoch, nrow=6, normalize=False)
    return running_results 

#%% 测试函数
def test(epoch):
    vae.eval()
    out_path = images_out_path+'/testing_results/' 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    with torch.no_grad():    
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        for i, (masked_img, img_batch) in enumerate(test_dataloader):        
            imgs = Variable(img_batch.type(FloatTensor))
            masked_img = Variable(masked_img.type(FloatTensor))
            batch_size = imgs.size(0)
            valing_results['batch_sizes'] += batch_size
            # reconstruct image    
            x_tilde_rec, _, _ = vae(imgs)
            # generate image
            #z = torch.randn(6, Z_DIM, 1, 1).cuda()
            #x_tilde_gen= model.decoder(z)
            # recover image
            x_tilde_gen, _, _ = vae(masked_img)
            
            batch_mse = ((x_tilde_gen - imgs) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(x_tilde_gen, imgs).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * math.log10((imgs.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        
        # Save sample
    x_cat = torch.cat([imgs[:6], masked_img[:6], x_tilde_rec[:6],x_tilde_gen[:6]], 0)
    sample = (x_cat.cpu().data + 1) / 2
    save_image(sample, out_path+"test_epoch_%d.png" % epoch, nrow=6, normalize=False)   
    return valing_results


#%% main
if __name__ == '__main__':
    #  一些设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")  
    parser.add_argument("--dataset_name", type=str, default="prepared_image/img_align_celeba_crop", help="name of the dataset")
    parser.add_argument("--dataset_train", type=str, default="prepared_image/CelebA_train_delete.txt", help="name of the train dataset label")
    parser.add_argument("--dataset_test", type=str, default="prepared_image/CelebA_test.txt", help="name of the test dataset label")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")            
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=64, help="size of random mask") 
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--img_root_path', type=str, default='E:\data', help='model version')     
    parser.add_argument('--version', type=str, default='1.1', help='model version') 
    opt = parser.parse_args(args=[])

    cuda = True if torch.cuda.is_available() else False
      
    images_out_path = 'output/images_stage1_v%s'% opt.version
    model_out_path = 'output/checkpoint_stage1_v%s'% opt.version
    statictics_out_path = 'output/statictics_stage1/'
    os.makedirs(images_out_path, exist_ok=True)
    os.makedirs(model_out_path, exist_ok=True)
    os.makedirs(statictics_out_path, exist_ok=True)
    
    set_seed(123)
    #  准备数据
    # Configure data loader          
    dataloader = DataLoader(
        ImageDataset(opt.img_root_path, opt.dataset_train, opt.dataset_name, 'prepared_image/gallery_list.txt',
                     img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        )

    test_dataloader = DataLoader(
        ImageDataset(opt.img_root_path, opt.dataset_test, opt.dataset_name, 'prepared_image/gallery_list.txt',
                     img_size=opt.img_size, mode="test"),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        )

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    DIM = 256
    Z_DIM = 512   
    LR = opt.lr

    # Losses for VAE
    #Reconstruction loss
    Recons = torch.nn.MSELoss(reduction='sum')  
    # Initialize VAE
    vae = VAE(opt.channels, DIM, Z_DIM)

    # load model checkpoint
    if opt.epoch != 0:
        # Load pretrained models
        # load model checkpoint
        pretrained_dict=torch.load("output/checkpoint_stage1_v%s/VAE_%d.pth" % (opt.version, opt.epoch)) 
        vae.load_state_dict(pretrained_dict)

    if cuda:
        vae.cuda()
        Recons.cuda()
       
    optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) 

    BEST_MSE = 99999
    LAST_SAVED = -1
    results = { 'img_Loss': [], 'z_loss': [], 'vae_loss': [], 'psnr': [], 'ssim': []}    
    for epoch in range(opt.epoch, opt.n_epochs):  
        #set_seed(epoch)        
        running_results = train(epoch)
        valing_results = test(epoch)
        
        cur_mse = valing_results['mse']
        if cur_mse <= BEST_MSE:
            BEST_SSIM = cur_mse
            LAST_SAVED = epoch
            print("Saving model! epoch: %d"%epoch) 
            torch.save(vae.state_dict(), "output/checkpoint_stage1_v%s/vae_%d.pth" % (opt.version, epoch)) 
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED)) 
       
        results['img_Loss'].append(running_results['img_Loss'] / running_results['batch_sizes'])
        results['z_loss'].append(running_results['z_loss'] / running_results['batch_sizes'])
        results['vae_loss'].append(running_results['vae_loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])       
        if epoch != 0 and epoch % 10 == 0:
            
            data_frame = pd.DataFrame(
                data={'img_Loss': results['img_Loss'], 'z_loss': results['z_loss'], 'vae_loss': results['vae_loss'],
                     'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(statictics_out_path + 'srf_train_results.csv', index_label='Epoch')

        
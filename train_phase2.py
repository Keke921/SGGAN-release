# -*- coding: utf-8 -*-
import argparse
import os
import math

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader

import pytorch_ssim
import pandas as pd

from datasets import *
from model.VAE import *

from utils import add_sn
from loss import GeneratorLoss, DiscriminatorLoss

#%%
def set_seed(seed):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

#%% stage 2 train    
def train_gan(epoch): 
    dis.train()
    gen.train()
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    out_path = images_out_path+'/training_results/' 
    if not os.path.exists(out_path):
        os.makedirs(out_path)    
        
    for i, (mask_img, img_batch) in enumerate(dataloader):
        #gen.train
        batches_done = epoch * len(dataloader) + i
        batch_size = img_batch.size(0)
        # Configure input   
        mask_img = Variable(mask_img.type(FloatTensor))  
        imgs = Variable(img_batch.type(FloatTensor))       
        running_results['batch_sizes'] += batch_size 
        del img_batch
        
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        # Adversarial ground truths
        valid = []
        fake = []
        for idx in range(3):
            valid_temp = Variable(FloatTensor(imgs.shape[0],1,16//(4**idx),16//(4**idx)).fill_(1.0), requires_grad=False)
            valid.append(valid_temp)
            fake_temp = Variable(FloatTensor(imgs.shape[0],1,16//(4**idx),16//(4**idx)).fill_(0.0), requires_grad=False)
            fake.append(fake_temp)
        del valid_temp, fake_temp
        with torch.no_grad():
            x_tilde, Lz,_ = vae(mask_img)
        fake_img = gen(x_tilde, mask_img)        
        # -----------------
        #  Train Gen
        # ----------------- 
        with torch.no_grad():    
            fake_pre = dis(fake_img)
        g_loss  = Gen_Loss(fake_img,imgs.detach(),fake_pre,valid)          
        g_loss.backward()                
        optimizerG.step()
        torch.cuda.empty_cache() 
        
        # -----------------
        #  Train Dis
        # -----------------       
        fake_pre = dis(fake_img.detach())
        real_pre = dis(imgs)
        
        d_real_loss = Dis_Loss(real_pre, valid)
        d_fake_loss = Dis_Loss(fake_pre, fake)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizerD.step()
        torch.cuda.empty_cache() 
        
        # loss for current batch before optimization 
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += d_real_loss.mean().item() * batch_size
        running_results['g_score'] += d_fake_loss.mean().item() * batch_size
        
        #  Log Progress
        if (batches_done % 300) == 0:#if batches_done > 0 and (batches_done % 300) == 0:
            f = open("output/train_loss2.txt", "a")  
            print("[Epoch %d/%d] [Batch %d/%d] [Gen Loss: %f] [Dis loss: %f]" 
               % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item(),d_loss.item()), file = f ) 
            f.close() 
        del d_loss, g_loss
    
    img_grid = torch.cat((mask_img[:6], imgs[:6], fake_img[:6]), 0)  
    img_grid = (img_grid.cpu().data + 1) / 2     
    save_image(img_grid, out_path+"/train_epoch_%d.png" % epoch, nrow=6, normalize=False)        

    return running_results    
    #test(epoch, vae, gen)      
    #torch.save(gen.state_dict(), "output/checkpoint%s/gen_%d.pth" % (opt.version, epoch)) 
    #torch.save(dis.state_dict(), "output/checkpoint%s/dis_%d.pth" % (opt.version, epoch)) 
#%% stage 2 train   
def test(epoch, vae, gen):
    gen.eval()
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
            x_tilde, _, _ = vae(masked_img)
            # recover image
            fake_img = gen(x_tilde, masked_img) 
            
            batch_mse = ((fake_img - imgs) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(fake_img, imgs).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * math.log10((imgs.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
    
    # Save sample        
    x_cat = torch.cat([imgs[:6], masked_img[:6], fake_img[:6]], 0)
    sample = (x_cat.cpu().data + 1) / 2
   
    save_image(sample,  out_path+"test_epoch_%d.png" % epoch, nrow=6, normalize=False)

    return valing_results
#%%
if __name__ == '__main__':
    #  一些设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=30, help="size of the batches")  
    parser.add_argument("--dataset_name", type=str, default="prepared_image/img_align_celeba_crop", help="name of the dataset")
    parser.add_argument("--dataset_train", type=str, default="prepared_image/CelebA_train_delete.txt", help="name of the train dataset label")
    parser.add_argument("--dataset_test", type=str, default="prepared_image/CelebA_test.txt", help="name of the test dataset label")
    parser.add_argument("--dataset_label", type=str, default="prepared_image/identity_CelebA.txt", help="name of the dataset label")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--num_classes", type=int, default=10177, help="number of the clesses")              
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--img_root_path', type=str, default='E:\data', help='model version')  
    parser.add_argument('--version', type=str, default='1.1', help='model version') 
    parser.add_argument('--stage1-path', type=str, default="./output/vae.pth", help='model version')     
    opt = parser.parse_args(args=[])

    cuda = True if torch.cuda.is_available() else False
    
    
    images_out_path = 'output/images_stage2_v%s/'% opt.version
    model_out_path = 'output/checkpoint_stage2_v%s/'% opt.version
    statictics_out_path = 'output/statictics_stage2_v%s/'% opt.version
    os.makedirs(images_out_path, exist_ok=True)
    os.makedirs(model_out_path, exist_ok=True)
    os.makedirs(statictics_out_path, exist_ok=True)
    
    
    #  准备数据
    set_seed(123)
    # Configure data loader          
    dataloader = DataLoader(
        ImageDataset( opt.img_root_path, opt.dataset_train, opt.dataset_name, 
                     'prepared_image/gallery_list.txt', 
                     img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        )

    test_dataloader = DataLoader(
        ImageDataset(opt.img_root_path, opt.dataset_test, opt.dataset_name, 
                     'prepared_image/gallery_list.txt',
                     img_size=opt.img_size,mode="test"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        )

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    DIM = 256
    Z_DIM = 512   
    LR = opt.lr

    # Losses 
    Dis_Loss = DiscriminatorLoss()
    Gen_Loss = GeneratorLoss()
    Recons = torch.nn.MSELoss()
    # Initialize model
    vae = VAE(opt.channels, DIM, Z_DIM)
    gen = Gen(opt.channels, DIM)
    dis = Dis(opt.channels, DIM,Z_DIM)
   
    pretrained_dict=torch.load(opt.stage1_path) 
    vae.load_state_dict(pretrained_dict)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    if opt.epoch != 0:
        # Load pretrained models
        # load model checkpoint
        gen.load_state_dict(torch.load("output/checkpoint_stage2_v%s/gen_%d.pth" % (opt.version, opt.epoch)))
        dis.load_state_dict(torch.load("output/checkpoint_stage2_v%s/dis_%d.pth" % (opt.version, opt.epoch)))
            
    if cuda:
        vae.cuda()
        gen.cuda()
        dis.cuda()
        Dis_Loss.cuda()
        Gen_Loss.cuda()
        Recons.cuda()
       
    optimizerG = torch.optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = torch.optim.Adam(dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    BEST_SSIM = 0
    LAST_SAVED = -1
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}    
    for epoch in range(opt.epoch, opt.n_epochs):
        running_results = train_gan(epoch)
        valing_results = test(epoch, vae, gen)           
        cur_ssim = valing_results['ssim']
        if cur_ssim >=  BEST_SSIM:
            BEST_SSIM = cur_ssim
            LAST_SAVED = epoch
            print("Saving model! epoch: %d"%epoch) 
            torch.save(gen.state_dict(), model_out_path + "gen_%d.pth" % epoch) 
            torch.save(dis.state_dict(), model_out_path + "dis_%d.pth" % epoch) 
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED)) 
       
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])         
        if epoch % 10 == 0: #if epoch != 0 and epoch % 10 == 0:           
            data_frame = pd.DataFrame(
                data={'d_loss': results['d_loss'], 'g_loss': results['g_loss'], 
                      'd_score': results['d_score'],'g_score': results['g_score'],
                     'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(opt.epoch, epoch + 1))
            data_frame.to_csv(statictics_out_path + 'srf_train_results.csv', index_label='Epoch')
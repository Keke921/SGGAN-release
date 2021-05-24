# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from utils import *


class Encoder(nn.Module):  #input image size: [128, 128, 3]
    '''
    initial: dim, dimension of features (top) ; z_dim, dimension of z
                mode: imput image type
    input: complete face images of partial face images. 
    output: mu, log_var, statistics of z1;
                xs, features of 3 levels
    '''
    def __init__(self,input_dim, dim, z_dim):
        super(Encoder,self).__init__()       
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential( downsample(input_dim, dim//8, stride=2), ResBlock(dim//8), #64    3
                           downsample(dim//8, dim//4, stride=2), 
                           #spatial_pyramid_pool(dim//4, 32, out_pool_size = [24,16,8], with_gp=False)
                           ),                                                         #32    7                   
            nn.Sequential( downsample(dim//4, dim//4, stride=2), ResBlock(dim//4),    #16    15
                           downsample(dim//4, dim//2, stride=2), ),                   #8     31
            nn.Sequential( downsample(dim//2, dim//2, stride=2), ResBlock(dim//2),    #4     63
                           downsample(dim//2, dim, stride=2)),                        #2     127
        ])
               
        self.condition_x = nn.Sequential(                   
            #ResBlock(dim),
            #nn.Conv2d(dim, z_dim * 2, 1, 1, 0),
            #spatial_pyramid_pool(dim, base_size = 4, out_pool_size = [3,2,1], with_gp=False),
            nn.Conv2d(dim, z_dim, 2, 1, 0),
            nn.BatchNorm2d(z_dim), Swish(),
            nn.Conv2d(z_dim, z_dim * 2, 1, 1, 0),
            )
        
        self.apply(weights_init)
        
    def forward(self, x):
        xs = []
        last_x = x
        for e in self.encoder_blocks:
            x = e(x)
            last_x = x
            xs.append(x)
        
        mu, log_var = self.condition_x(last_x).chunk(2, dim=1)
        #mu = torch.flatten(mu, start_dim=1)
        #log_var = torch.flatten(log_var, start_dim=1)
        xs.reverse()
        return mu, log_var, xs
    
    
if __name__=='__main__':    
    #test encoder
    x = torch.ones([3,3,28,28])
    x2 = torch.ones([32,3,128,64])
    encoder1 = Encoder(3, 256, 512)
    mu, log_var, xs = encoder1(x)  
    mu2, log_var2, xs2 = encoder1(x) 
    
    
    
    encoder2 = Encoder(32, 3, 256, 512)
    mu, log_var, xs = encoder1(x2)  
    
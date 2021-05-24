# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
#from torch.distributions.normal import Normal
#from torch.distributions import kl_divergence

from utils import *
from loss import kl
#

class Decoder(nn.Module):  #input image size: [128, 128, 3]
#    
#    initial: input_dim, channels of input image; 
#             dim, dimension of features (top) ; z_dim, dimension of z
#             
#    input: z, sample from encoder's output; xs, features of diffetent levels. 
#    output: x_tilde, reconstructed image;
#            kl_div, 
#    
    def __init__(self, input_dim, dim, z_dim):
        super(Decoder,self).__init__()
        self.dim = dim
        self.sample_from_z = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 2, 1, 0),
            nn.BatchNorm2d(dim),
            Swish(),)

        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(upsample(dim, dim // 2), ResBlock(dim // 2), upsample(dim // 2, dim // 2),), #8
            nn.Sequential(upsample(dim // 2 , dim // 4), ResBlock(dim // 4), upsample(dim // 4, dim // 4), ), #32
            nn.Sequential(upsample(dim // 4, dim // 4),
                          ResBlock(dim // 4),
                          nn.Upsample(scale_factor=2),
                          nn.Conv2d(dim // 4, input_dim, 3, 1, 1),
                          nn.Tanh()),  # 64 h3
             ])

        self.apply(weights_init)
        
    def forward(self, z):        
        h_dec = self.sample_from_z(z)   #2*2
        for d in self.decoder_blocks:
            h_dec = d(h_dec)
        x_tilde = h_dec
        return x_tilde
         
if __name__=='__main__':    
    #test encoder
    from encoder import Encoder
    x = torch.ones([32,3,128,128])
    
    encoder1 = Encoder(32, 3, 256, 512)
    mu, log_var, xs = encoder1(x)  
    z = reparameterize(mu, torch.exp(0.5 * log_var)) 
    decoder = Decoder(3, 256, 512)
    x_tilde, kl_div = decoder(z,xs)  
    x_tilde, kl_div = decoder(z,xs,multi_level = False) 
    x_tilde2, kl_div2 = decoder(z) 

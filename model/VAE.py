import torch
import torch.nn as nn
#from torch.distributions.normal import Normal
#from torch.distributions import kl_divergence

from utils import *
from loss import kl#VAELoss
from model.encoder import Encoder
from model.decoder import Decoder

class VAE(nn.Module):

    def __init__(self,  input_dim, dim, z_dim):
        super(VAE,self).__init__()

        self.encoder = Encoder(input_dim, dim, z_dim)
        self.decoder = Decoder(input_dim, dim, z_dim)
        #self.apply(add_sn)
        self.apply(weights_init)
        
    def forward(self, x, multi_level=True):
        """
        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """
        mu, log_var, xs = self.encoder(x)

        # (B, z_dim, img_size//8, img_size//8)
        z = reparameterize(mu, torch.exp(0.5 * log_var)) #Normal(mu, log_var.mul(.5).exp()).rsample()    
        x_tilde = self.decoder(z) #        
        #vae_loss = VAELoss(x_tilde, x, mu, log_var, losses)
        kl_div = kl(mu, log_var)
        return x_tilde, kl_div, xs
    

class Gen(nn.Module):
    def __init__(self,  input_dim, dim):
        super(Gen,self).__init__()
        
        self.fuser = nn.Sequential( nn.Conv2d(input_dim*2, dim//4, 3,1,1))
        
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential( nn.Conv2d(dim//4, dim//4, kernel_size=3, stride=2, padding=1),
                           nn.BatchNorm2d(dim//4), Swish(), ResBlock(dim//4),                          #64  3
                           hybrid_dilated_convolution(dim//4, dim//4, with_gp=True) ),                 #64  13
            nn.Sequential( downsample(dim//4, dim//2), ResBlock(dim//2),                               #32  27
                           hybrid_dilated_convolution(dim//2, dim//2, with_gp=True),),                 #32  37          
            nn.Sequential( downsample(dim//2, dim//2), ResBlock(dim//2), 
                           downsample(dim//2, dim),
                           nn.Conv2d(dim, dim, 3, 1, 1),nn.LeakyReLU(),  )    #8   151
        ])
        
        '''
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(upsample(dim, dim// 2), ResBlock(dim// 2), upsample(dim// 2, dim // 2), ),   #32                                            #32
            nn.Sequential(upsample(dim // 2*2 , dim // 2), ResBlock(dim // 2), ),                      #64
            nn.Sequential(upsample(dim // 4*3, dim // 4),ResBlock(dim // 4),
                          nn.Conv2d(dim // 4, dim // 4, 3, 1, 1),nn.LeakyReLU(), 
                          nn.Conv2d(dim // 4, dim // 4, 3, 1, 1),nn.LeakyReLU(),
                          nn.Conv2d(dim // 4, input_dim, 1),
                          nn.Tanh()),  
             ])
        '''
        #0119 去掉skip-connection看一下有没有网格
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(upsample2(dim, dim// 2), ResBlock(dim// 2), upsample2(dim// 2, dim // 2),
                          nn.Conv2d(dim // 2, dim // 2, 3, 1, 1),nn.LeakyReLU(), ),   #32                                            #32
            nn.Sequential(upsample2(dim // 2*2 , dim // 2), ResBlock(dim // 2), 
                          nn.Conv2d(dim // 2, dim // 2, 3, 1, 1),nn.LeakyReLU(), ),                      #64
            nn.Sequential(upsample2(dim // 2, dim // 4),ResBlock(dim // 4),
                          nn.Conv2d(dim // 4, dim // 4, 3, 1, 1),nn.LeakyReLU(), 
                          nn.Conv2d(dim // 4, dim // 4, 3, 1, 1),nn.LeakyReLU(),
                          nn.Conv2d(dim // 4, input_dim, 1),
                          nn.Tanh()),  
             ])
        self.apply(add_sn)
        self.apply(weights_init)
        
    def forward(self, x, x_tilde):
        x = self.fuser(torch.cat([x, x_tilde], dim=1))
        
        xs=[]
        for e in self.encoder_blocks:
            x = e(x)
            last_x = x
            xs.append(x)
        #last_x = self.hdc(last_x)       
        xs.reverse()
        
        h_dec = last_x
        for i in range(len(self.decoder_blocks)):  
            if i ==1:
                h_dec = torch.cat([h_dec, xs[i]], dim=1)
            h_dec =  self.decoder_blocks[i](h_dec)                    
        '''
        for e in self.encoder_blocks:
            x = e(x)
        h_dec = x
        for d in self.decoder_blocks:
            h_dec =  d(h_dec)
        '''       
        return h_dec

class Dis(nn.Module):
    def __init__(self,  input_dim, dim,z_dim):
        super(Dis,self).__init__()
        model = VAE(input_dim, dim, z_dim)
        #model.apply(add_sn)
        #model.load_state_dict(torch.load("output\\checkpoint_stage1_v1.1\\vae_135.pth"))
        self.encoder_blocks = model.encoder.encoder_blocks
        #self.encoder_blocks.eval()
        '''
        for param in self.encoder_blocks.parameters():
            param.requires_grad = False
        '''
        self.discriminator_blocks = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(16),
                          nn.Conv2d(dim//4, dim, kernel_size=1),
                          nn.BatchNorm2d(dim),  Swish(),
                          nn.Conv2d(dim, 1, kernel_size=1),
                          nn.Sigmoid()),                                       #32->16  RF:7
            nn.Sequential(nn.AdaptiveAvgPool2d(4),
                          nn.Conv2d(dim//2, dim, kernel_size=1),
                          nn.BatchNorm2d(dim), Swish(),
                          nn.Conv2d(dim, 1, kernel_size=1),                    #8->4    RF:31
                          nn.Sigmoid()),                                       
            nn.Sequential(nn.Conv2d(dim, dim, 2,1,0), 
                          nn.BatchNorm2d(dim), Swish(),  
                          nn.Conv2d(dim, 1, kernel_size=1),            
                          nn.Sigmoid()),                                       #2->1    RF:127
        ])

        self.apply(add_sn)
        self.apply(weights_init)
        
        '''
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential( downsample(input_dim, dim//4), ResBlock(dim//4), downsample(dim//4, dim//4) ),  #32
            nn.Sequential( downsample(dim//4, dim//2), ResBlock(dim//2),downsample(dim//2, dim//2)  ),     #8
            nn.Sequential( downsample(dim//2, dim), ResBlock(dim), downsample(dim, dim)  ),                #2
        ])
        '''
        '''
        self.discriminator = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(dim, 1024, kernel_size=1),
                                           Swish(),
                                           nn.Conv2d(1024, 1, kernel_size=1),
                                           nn.Sigmoid() )
        '''
        self.discriminator_blocks.apply(weights_init)
    
    def forward(self, x):       
        xs = []
        validities = []
        for e in self.encoder_blocks:
            x = e(x)   
            xs.append(x)
        
        for i in range(len(self.discriminator_blocks)):
           val =  self.discriminator_blocks[i](xs[i])  
           validities.append(val)
        return validities        
    
if __name__ == '__main__':
    vae = VAE(3, 256,512)
    img = torch.rand(8, 3, 128, 128)
    img2 = torch.rand(8, 3, 128, 128)
    img=(img-img.mean())/img.std()
    x_tilde,_,xs= vae(img)

    gen = Gen(3, 256,512)
    x = gen(img,img2)
    
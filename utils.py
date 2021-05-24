# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from light_cnn import LightCNN_29Layers_v2

import cv2
import numpy as np
import face_alignment
from numpy.linalg import inv, lstsq  #norm, 
from numpy.linalg import matrix_rank as rank
from PIL import Image
import skimage
import skimage.filters
#import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class downsample(nn.Module): 
    def __init__(self, in_channel, out_channel, stride = 2):  #stride can be 2 or 4
        super().__init__() 
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, (4-stride)//2),
            nn.BatchNorm2d(out_channel),
            Swish(), #nn.ReLU(True)
        )
   
    def forward(self, x):
        return self.layers(x)

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__() 
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*scale_factor**2, 3, 1, 1),
            nn.LeakyReLU(),
            nn.PixelShuffle(scale_factor),
        )
   
    def forward(self, x):
        return self.layers(x)

class upsample2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__() 
        
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
            #nn.ReLU(True)
        )
   
    def forward(self, x):
        return self.layers(x)

#
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prelu = nn.PReLU()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),Swish(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),Swish(),
        )

    def forward(self, x):
        return self.prelu(x + self.block(x))

def ConvBNReLU(in_chan, out_chan, ks, dilation=1, padding=0):
    block = nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel_size = ks, stride = 1,
                  padding = padding, dilation = dilation, bias = True),
        nn.BatchNorm2d(out_chan),
        Swish(),)
    return block

class hybrid_dilated_convolution(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True):
        super(hybrid_dilated_convolution, self).__init__()
        self.with_gp = with_gp

        self.conv1 = ConvBNReLU(in_chan, out_chan//4, ks=1, dilation=1, padding=0)  
        self.conv2 = ConvBNReLU(in_chan, out_chan//4, ks=3, dilation=1, padding=1)  #ks=5, dilation=1, padding=2
        self.conv3 = ConvBNReLU(in_chan, out_chan//4, ks=3, dilation=2, padding=2)  #ks=5, dilation=2, padding=4
        self.conv4 = ConvBNReLU(in_chan, out_chan//4, ks=3, dilation=5, padding=5) #ks=5, dilation=5, padding=10
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan//4, ks=1)
            self.conv_out = ConvBNReLU(out_chan//4*5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan, out_chan, ks=1)

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat


class spatial_pyramid_pool(nn.Module):   
    '''
    修改记录：
    0916: 1. AdaptiveMaxPool2d -> AdaptiveAvgPool2d，MaxPool 有很多一样的值。
          2. out_pool_size = [12,6,3] -> [12,9,4], 上采样方式变为： F.interpolate
    0917: out_pool_size = [9,6,3], 池化完上采样，卷积减少特征数量， 解释参考PSPNet
          添加with_gp=True，添加本身大小不变的图 （[bs,c,16,16]）
    0918: base_size调整为可变 （[batch_size, channels, base_size, base_size]）      
    '''
    def __init__(self, in_chan = 256, base_size = 16, out_pool_size = [9,6,3], with_gp=True):
    #out_pool_size: an int vector of expected output size of max pooling layer
        super().__init__()
        self.base_size = base_size
        self.out_pool_size = out_pool_size
        self.with_gp = with_gp
        
        self.conv = ConvBNReLU(in_chan, in_chan//(1+len(out_pool_size)), 3, 1, 1)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, in_chan, ks=1)
            self.conv_out = ConvBNReLU(in_chan+in_chan//(1+len(out_pool_size)), in_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(in_chan, in_chan, ks=1)
        
    def forward(self, previous_conv):
    #previous_conv: a tensor vector of previous convolution layer
    #returns: a tensor vector with shape [batch size x n x base_size x base_size]     
        bs = self.base_size
        spp = F.interpolate(previous_conv, (bs,bs), mode='bilinear', align_corners=True)
        spp = self.conv(spp)
        for i in range(len(self.out_pool_size)):
            AvgPool = nn.AdaptiveAvgPool2d(self.out_pool_size[i]) #Maxpool = nn.AdaptiveMaxPool2d(self.out_pool_size[i])
            x = AvgPool(previous_conv)            #x = Maxpool(previous_conv)            
            x = F.interpolate(x, (bs,bs), mode='bilinear', align_corners=True) 
            spp = torch.cat((spp,self.conv(x)), 1)            
        #print("spp output size",spp.size())
        if self.with_gp:
            avg = self.avg(previous_conv)
            x = self.conv1x1(avg)
            x = F.interpolate(x, (bs, bs), mode='bilinear', align_corners=True)
            spp = torch.cat((spp,self.conv(x)), 1) 
        return self.conv_out(spp)


class hybrid_spp(nn.Module):
    def __init__(self, in_chan=512, out_chan=512, base_size = 16, out_pool_size = [9,6,3], with_gp=True):
        super(hybrid_spp, self).__init__()
        self._seq = nn.Sequential(
            hybrid_dilated_convolution(in_chan = in_chan, out_chan=out_chan, with_gp=with_gp),
            spatial_pyramid_pool(in_chan = out_chan, base_size=base_size,out_pool_size=out_pool_size, with_gp=with_gp),
            )
        
    def forward(self, x):
        return self._seq(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class feature_extract(nn.Module):
    '''
    ref:
    we use lightCNN to extract fearuee. The code is from:
    https://github.com/AlfredXiangWu/LightCNN
    '''   
    def __init__(self):
        super().__init__()
        self.model = LightCNN_29Layers_v2(num_classes=80013)
        self.model.eval()
        self.model = torch.nn.DataParallel(self.model).cuda()
        checkpoint = torch.load('LightCNN_29Layers_V2_checkpoint.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False
        #self.model = model.modules
        
    def forward(self, x):
        #input     = torch.zeros([x.size(0), 1, 128, 128], device=x.device, dtype=x.dtype)
        #转为灰度图
        input = x[:,0]*0.3+x[:,1]*0.59 + x[:,2]*0.11
        input = input.unsqueeze(1)
        return self.model(input)   
        

def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z

def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m

#%%  测试时粗略对齐
class ImgAlignment(nn.Module):
    def __init__(self, img_size = (128,128), device='cpu'):
        super(ImgAlignment,self).__init__()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)    
        self.ref = np.array([
            [38,  55.5],
            [89,  54],
            [73,  79],
            [45,  104],
            [84,  104]], np.float32)
        self.img_size  = img_size
        
    def forward(self, img):       
        # Image 转成opencv格式。
        img = np.asarray(img)  
        preds = self.fa.get_landmarks_from_image(img)
        eye1 = preds[0][36:42]
        left_eye = (eye1.max(0)+eye1.min(0))/2
        eye2 = preds[0][42:48]
        right_eye = (eye2.max(0)+eye2.min(0))/2
        nose = preds[0][30]
        mouth = preds[0][48:60]
        mouth_left = mouth[mouth[:,0].argmin()]
        mouth_right = mouth[mouth[:,0].argmax()]
        landmark = np.array([left_eye,right_eye, nose,mouth_left,mouth_right])  
        
        similar_trans_matrix = findNonreflectiveSimilarity(landmark.astype(np.float32), self.ref)
        aligned_img = cv2.warpAffine(img.copy(), similar_trans_matrix, self.img_size)
        
        return Image.fromarray(aligned_img)        


def findNonreflectiveSimilarity(uv, xy, K=2): 
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
 
    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
 
    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
 
    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')
 
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
 
    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])
  
    T = inv(Tinv) 
    T[:, 2] = np.array([0, 0, 1]) 
    T = T[:, 0:2].T
    return T


if __name__=='__main__':    
    x1 = torch.ones([2,3,128,128])
    ds = downsample(3, 6, stride = 4)
    x_ds = ds(x1)
    
    #test hybrid_dilated_convolution
    x1 = torch.ones([2,3,128,128])
    hdc = hybrid_dilated_convolution(3,64)
    x = hdc(x1)
    print(x.shape)
    
    #test feature extracter
    x1 = torch.ones([2,3,128,128])
    fe = feature_extract()
    f = fe(x1)  #1*256
    
    #test spatial_pyramid_pool
    #spp = spatial_pyramid_pool(in_chan = 256)
    x1 = torch.ones([32,256,12,12])
     
    spp = hybrid_spp(256, 128, 12, [9,6,3],)
    x1_spp = spp(x1)  
    
    
    x1.shape
# -*- coding: utf-8 -*-
import torch.nn as nn
#import torch.nn.functional as F
import torch
from torch.distributions import kl_divergence
from utils import feature_extract
import math

def kl(mu, log_var):
    '''
    kl[q_z_x | p_z]  q_z_x and p_z are set as Gaussian distribution
    Parameters
    ----------
    mu; log_var 
        
    Returns
    -------
    kl divergence
    '''
    #p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
    #q_z_x = Normal(mu, logvar.mul(.5).exp())
    #kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1)
    return loss.mean()#torch.mean(loss, dim=0)



def kl_2(delta_mu, delta_log_var, mu, log_var):
    '''
    Parameters
    ----------
    delta_mu; delta_log_var; mu; log_var

    Returns
    -------
    kl divergence

    '''
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=1)
    return loss.mean() #torch.mean(loss, dim=0)

def VAELoss(x_tilde,x, mu, log_var, losses):
    L_x = nn.MSELoss(reduction='sum')(x_tilde,x)
    L_z = kl(mu, log_var)+  losses[0] +  losses[1]   #
    #print( "L_x %d, L_z %d, L_z1_1 %d, L_z1_2 %d " %(L_x, L_z, losses[0], losses[1]) )#, D_loss.
    return L_x+L_z


class FeatureDistance(nn.Module):
    def __init__(self):
        super().__init__() 
        self.loss = nn.CosineSimilarity(dim = 1, eps = 1e-6)
        
    def forward(self, feature_dot, feature_tar):
        BS = len(feature_dot)   #batch size
        distance = 1 - self.loss(feature_dot.view(BS,-1), feature_tar.view(BS,-1))
        return distance.sum()/BS

class ImageDistance(nn.Module):
    def __init__(self):
        super().__init__() 
        self.L1loss = nn.L1Loss() #nn.MSELoss(reduction='sum')
        #self.L2loss = nn.MSELoss()
    def forward(self, data, target):
        #BS = target.size(0)    #batch size
        return self.L1loss(data, target)# + 0.8*self.L2loss(data, target)

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
        
class GeneratorLoss(nn.Module):
    def __init__(self, weight = [100,50,1,0.5]):
        super().__init__()    
        self. weight = weight
        self.feature_loss = FeatureDistance()
        self.imagedistance = ImageDistance()
        self.F = feature_extract()
        self.adv_loss = DiscriminatorLoss()
        self.tv_loss = TVLoss()
    def forward(self, data, target, data_pre, data_label):
        ReconL =   self.imagedistance(data,target)    
        FeaL = self.feature_loss(self.F(data),self.F(target))
        AdvL = self.adv_loss(data_pre, data_label)
        TVL = self.tv_loss(data)
        return self.weight[0]*ReconL + self.weight[1]*FeaL +  self.weight[2] * AdvL + self.weight[3]*TVL

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
      
    def forward(self, pre, val):   
        
        small_rf = self.loss(pre[0],val[0])
        median_rf = self.loss(val[1],val[1])
        big_rf = self.loss(val[2],val[2])
        
        weight_sum = math.exp(small_rf) + math.exp(median_rf) + math.exp(big_rf)
        
        weight_small = math.exp(small_rf)/weight_sum
        weight_median = math.exp(median_rf)/weight_sum
        weight_big = math.exp(big_rf)/weight_sum
        
        return weight_small*small_rf+weight_median*median_rf+weight_big*big_rf
        
   


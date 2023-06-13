import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

from torch.autograd import Variable

import os
import sys
import copy
import math
import numpy as np
from typing import List

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


from torch.autograd import Variable

class LearnedPositionEncoding1(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1, input_size = 7):
        super().__init__(d_model, input_size*input_size)
        self.input_size = input_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.view(self.d_model, self.input_size, self.input_size, ).unsqueeze(0)
        #print(x.shape, weight.shape)
        x = x + weight
        return self.dropout(x)


class LearnedPositionEncoding2(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1, input_size = 7):
        super().__init__(input_size*input_size, d_model)
        self.input_size = input_size
        self.d_model = d_model
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.view(self.input_size * self.input_size, self.d_model).unsqueeze(1)
        x = x + weight
        return self.dropout(x)


class MultiLevelTransformer(nn.Module):
    def __init__(self, args, input_dim: int = 64*7, patch_size: List[int] = [4,4,4], d_model: int = 512, 
                nhead: int = 8, num_encoder_layers: List[int] = [4,4,4],
                num_decoder_layers: List[int] = [4,4,4], dim_feedforward: int = 2048, dropout: float = 0.1):
        super(MultiLevelTransformer, self).__init__()
        self.zeros = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=False)
        self.d_model = d_model
        self.patch_size = patch_size

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.global_position_embedding = LearnedPositionEncoding1(d_model = d_model, dropout = dropout, input_size = np.prod(patch_size))
        self.position_embedding = nn.ModuleList([
            LearnedPositionEncoding2(d_model = d_model, dropout = dropout, input_size = patch_size[i]) \
                for i in range(len(patch_size))])

        self.query_embedding = nn.ModuleList([
            nn.Embedding(patch_size[i] * patch_size[i], d_model) \
                for i in range(len(patch_size))])

        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, 
                        dim_feedforward = dim_feedforward, dropout = dropout, activation = args.activation)
                        , num_encoder_layers[i], encoder_norm) for i in range(len(patch_size))])

        self.decoder = nn.ModuleList([
            nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model = d_model, nhead = nhead, 
                        dim_feedforward = dim_feedforward, dropout = dropout, activation = args.activation)
                        , num_decoder_layers[i], decoder_norm) for i in range(len(patch_size))])

        self.bottle_neck = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model//4), nn.LayerNorm(d_model//4), nn.GELU(),
            nn.Linear(d_model//4, d_model), nn.LayerNorm(d_model)) 
            for i in range(len(patch_size))])

        
        self.pre_conv = nn.Conv2d(input_dim, d_model, kernel_size=1, bias=False)
        self.final_layer1 = nn.Conv2d(d_model, input_dim, kernel_size=1, bias=False)
        self.final_layer2 = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model, d_model // 2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model // 2),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model // 2, d_model // 4, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model // 4),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model // 4, d_model // 8, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(d_model // 8),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(d_model // 8, 1, kernel_size=1, bias=False))
            
    def calculate_size(self, level):
        S = self.patch_size[level]
        P = 1
        for i in range(level+1, len(self.patch_size), 1):
            P *= self.patch_size[i]
        return P, S

    def forwardDOWN(self, x, encoder_block, position_embedding, level):
        _, BPSPS, C = x.size()
        P, S = self.calculate_size(level)
        B = BPSPS // (P*S*P*S)
        x = x.view(B, P, S, P, S, C).permute(2,4,0,1,3,5).contiguous().view(S*S, B*P*P, C) #(SS, BPP, C)
        pad = self.zeros.expand(-1, B*P*P, -1)
        x = encoder_block(src = torch.cat((pad.detach(), position_embedding(x)), dim=0))

        latent_patch = x[0,:,:].unsqueeze(0).contiguous() #(1, BPP, C)
        latent_pixel = x[1:,:,:].contiguous() #(SS, BPP, C)
        #print(x.size())

        return latent_patch, latent_pixel


    def forwardUP(self, latent_patch, latent_pixel, decoder_block, query, level):
        SS, BPP, C = latent_pixel.size()
        #1, BPP, C = latent_patch.size()
        P, S = self.calculate_size(level)
        B = BPP // (P*P)
        latent = torch.cat((latent_patch, latent_pixel),dim=0)
        out = decoder_block(memory = latent, tgt = query.weight.unsqueeze(1).expand(-1, BPP, -1) ) #(SS, BPP, C)
        out = out.view(S, S, B, P, P, C).permute(2,3,0,4,1,5).contiguous().view(1, B*P*S*P*S, C) #(1, BSPSP, C)
        return out
        
    
    def forward(self, x):

        x = self.pre_conv(x)
        B, C, H, W = x.size()  #(B, C, H, W)

        x = self.global_position_embedding(x)
        x = x.permute(0,2,3,1).contiguous().view(B*H*W,C).unsqueeze(0) #(1, BHW, C)
        latent_list = []
        for i in range(len(self.encoder)):
            x, l = self.forwardDOWN(x=x, encoder_block=self.encoder[i], position_embedding=self.position_embedding[i], level=i)
            latent_list.append(self.bottle_neck[i](l))
        for i in range(len(self.encoder)-1, -1, -1):
            x = self.forwardUP(latent_patch=x, latent_pixel=latent_list[i], decoder_block=self.decoder[i], query=self.query_embedding[i], level=i)
        x = x.squeeze(0).view(B, H, W, C).permute(0,3,1,2).contiguous()
        #out = self.final_layer1(x)
        return self.final_layer1(x), self.final_layer2(x.detach())


def Create_nets(args):
    
    dim = 64*7
    transformer = MultiLevelTransformer(args)
    transformer.apply(weights_init_normal)


    return transformer


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
swin_transformer input:192*32*32
styTR-2 position encoding+3 residual connection+laplas
return y_fake yhr_fake ylr_fake
gate using cross window attention

@Date     :2022/08/31 09:46:47
@Author      :Yifan Li
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .swin_transformer_networks import SwinTransformer, CrossSwinTransformerBlock
from PIL import Image
import torchvision.transforms as transforms


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_layer=1):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_layer = num_layer
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        # device = torch.device(deviceStr)
        # kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1],
                       x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1],
                       x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel.to(x.device))

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_layer):
            filtered = self.conv_gauss(current, self.kernel.to(img.device))
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(
                    up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.InstanceNorm2d(in_features),    
        )

    def forward(self, x):
        return x + self.block(x)

####################G########################
class refine(nn.Module):
    def __init__(self, refine_num=2, depth=2, num_heads=4, embed_dim=48, window_size=8,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(refine, self).__init__()
        self.refine_num = refine_num
        self.depth = depth
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, 3, 2, padding=1),
            nn.InstanceNorm2d(48),
            nn.LeakyReLU(),
        )
        crossSwin = [
            CrossSwinTransformerBlock(dim=embed_dim,  # input_resolution=input_resolution,
                                      num_heads=num_heads,
                                      window_size=window_size,
                                      shift_size=0 if (
                                          i % 2 == 0) else window_size // 2,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(
                                          drop_path, list) else drop_path,
                                      norm_layer=norm_layer)
            for i in range(depth)]
        self.deConv = nn.Sequential(
            nn.Conv2d(96, 48, 1, 1, padding=0),  # no in
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ResidualBlock(48),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Tanh()
        )
        self.deConv_hr = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Tanh()
        )
        self.crossSwin = nn.Sequential(*crossSwin)
        self.norm = norm_layer(embed_dim)

    def forward(self, x, y):  # x hr 256 256 3 y lr 128 128 48
        x = self.conv(x)  # 128 128 48
        hr = x
        b, c, h, w = x.shape
        x_fla = x.flatten(2).transpose(1, 2)  # b h*w c
        for j in range(self.refine_num):
            hr_fla = hr.flatten(2).transpose(1, 2)
            y_fla = y.flatten(2).transpose(1, 2)  # b h*w c
            for i in range(self.depth):
                hr_fla = self.crossSwin[i]((x_fla, y_fla), h, w)
            hr_fla = self.norm(hr_fla)
            hr = hr_fla.transpose(1, 2).reshape(b, c, h, w)
            y = y + hr
        if self.refine_num == 0:
             y = y + hr
        y = self.deConv(torch.cat((x, y), 1))
        hr = self.deConv_hr(hr) 
        return y, hr


class SwinUnitLow(nn.Module):
    def __init__(self, input_dim=3, gf_dim=48, n_embd=192):
        super(SwinUnitLow, self).__init__()
        # inputs shape * 3 * 256 * 256

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, gf_dim, 7, 1),  # dim*256*256
            nn.InstanceNorm2d(gf_dim),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(gf_dim, gf_dim * 2, 3, 2, padding=1),  # 2dim*128*128
            nn.InstanceNorm2d(gf_dim * 2),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, 2, padding=1),  # 4dim*64*64
            nn.InstanceNorm2d(gf_dim * 4),
            nn.LeakyReLU(),
        )
        # position encoding[HW,C]
        self.pos_emb = nn.Conv2d(gf_dim*4, gf_dim*4, (1, 1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

        self.drop = nn.Dropout(0.0)

        # swin transformer block

        self.blocks = [SwinTransformer()]
        self.blocks = nn.Sequential(*self.blocks)

        self.deConv3 = nn.Sequential(
            nn.Conv2d(gf_dim*8, gf_dim*4, 1, 1, padding=0),
            nn.InstanceNorm2d(gf_dim * 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, 1, padding=1),  # 2dim
            nn.InstanceNorm2d(gf_dim * 2),
            nn.LeakyReLU(),
        )
                         
        self.deConv2 = nn.Sequential(
            nn.Conv2d(gf_dim*4, gf_dim*2, 1, 1, padding=0),
            nn.InstanceNorm2d(gf_dim * 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(gf_dim * 2, gf_dim * 1, 3, 1, padding=1),  # dim
            nn.InstanceNorm2d(gf_dim),
            nn.LeakyReLU(),
        )

        self.deConv1 = nn.Sequential(
            nn.Conv2d(gf_dim*2, gf_dim*1, 1, 1, padding=0),
            nn.InstanceNorm2d(gf_dim),
            nn.LeakyReLU(),
        )

        self.deConv0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(gf_dim, input_dim, 7, 1),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.conv1(x)  # dim 128
        x_conv1 = x
        x = self.conv2(x)  # 2dim 64
        x_conv2 = x
        x = self.conv3(x)  # 4dim 32
        x_conv3 = x
        [b, c, h, w] = x.shape

        # position embedding in stytr2
        x_pool = self.averagepooling(x)  # b c 32 32
        pos = self.pos_emb(x_pool)  # b c 18 18
        position_embeddings = F.interpolate(pos, mode='bilinear',
                                            size=x.shape[-2:])  # each position maps to a (learnable) vector b c h w

        x = x.view(b, c, h * w).transpose(1, 2).contiguous()  # b hw c
        position_embeddings = position_embeddings.view(
            b, c, h * w).transpose(1, 2).contiguous()

        x = self.drop(x + position_embeddings)  # [b,hw,c]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        #########################
        x = self.blocks(x)  # b c h w 4dim 32 32
        #########################

        # x=self.deConv4(x)#4dim 64 64

        x = torch.cat((x, x_conv3), 1)  # 8dim
        x = self.deConv3(x)  # 2dim 64 64

        x = torch.cat((x, x_conv2), 1)  # 4dim
        x = self.deConv2(x)  # 1dim 128 128

        x = torch.cat((x, x_conv1), 1)  # 2dim
        x_1dim = self.deConv1(x)  # 1dim 128 128

        x = self.deConv0(x_1dim)

        return x, x_1dim


class SwinUnit(nn.Module):
    def __init__(self, refine_num = 2):
        super(SwinUnit, self).__init__()
        self.lap_pyramid = Lap_Pyramid_Conv()
        self.generator_low = SwinUnitLow()
        self.refine = refine(refine_num)
        self.refine_num = refine_num

    def forward(self, x):
        # pyr0:high_res256 pyr1:low_res128
        xhr, xlr = self.lap_pyramid.pyramid_decom(img=x)
        ylr_fake, ylr_fake_1dim = self.generator_low(xlr)
        y, hr = self.refine(xhr, ylr_fake_1dim)
        return y, xlr, ylr_fake  # y_fake ylr_fake


####################D########################
class HrDiscriminator(nn.Module):
    def __init__(self, input_dim=3, df_dim=64):
        super(HrDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, df_dim, 4, 2, padding=1),  # w/2
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(df_dim, df_dim * 2, 4, 2, padding=1), # w/4
            nn.InstanceNorm2d(df_dim * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, padding=1), # w/8
            nn.InstanceNorm2d(df_dim * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, padding=1), # w/16
            nn.InstanceNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, True),

            nn.ZeroPad2d((2, 1, 2, 1)),
            nn.Conv2d(df_dim * 8, 1, 4, 1)
        )

    def forward(self, inputs):
        return self.layers(inputs)

class LrDiscriminator(nn.Module):
    def __init__(self, input_dim=3, df_dim=64):
        super(LrDiscriminator, self).__init__()
        self.layers =  nn.Sequential(
            nn.Conv2d(input_dim, df_dim * 2, 4, 2, padding=1),
            nn.InstanceNorm2d(df_dim * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, padding=1),
            nn.InstanceNorm2d(df_dim * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, padding=1),
            nn.InstanceNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, True),

            nn.ZeroPad2d((2, 1, 2, 1)),
            nn.Conv2d(df_dim * 8, 1, 4, 1)
        )
    def forward(self, inputs):
        return self.layers(inputs)


class MsDis(nn.Module):
    def __init__(self, num_scales=1, input_dim=3, df_dim=64):
        super(MsDis, self).__init__()
        self.input_dim = input_dim
        self.df_dim = df_dim
        self.lap = Lap_Pyramid_Conv()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.lrdis = nn.ModuleList()
        self.hrdis = HrDiscriminator()
        for i in range(num_scales):
            self.lrdis.append(LrDiscriminator())
    
    def forward(self, x, x_lr):
        # hr, lr = self.lap.pyramid_decom(x)
        outputs=[]
        # outputs.append(self.hrdis(hr))
        # for dis in self.lrdis:
        #     outputs.append(dis(lr))
        #     lr = self.downsample(lr)
        outputs.append(self.hrdis(x))
        for dis in self.lrdis:
            outputs.append(dis(x_lr))
            x_lr = self.downsample(x_lr)
        return outputs

    def dis_loss(self, real, fake):
        fake_output = self.forward(fake)
        real_output = self.forward(real)
        loss = 0
        for i in range(len(fake_output)):
            loss += torch.mean((fake_output[i]-0)**2) + torch.mean((real_output[i]-1)**2) 
        return loss

    def dis_loss2(self, real, real_lr, fake, fake_lr):
        fake_output = self.forward(fake, fake_lr)
        real_output = self.forward(real, real_lr)
        loss = 0
        for i in range(len(fake_output)):
            loss += torch.mean((fake_output[i]-0)**2) + torch.mean((real_output[i]-1)**2) 
        return loss

    def gen_loss(self, fake):
        fake_output = self.forward(fake)
        loss = 0
        for i in range(len(fake_output)):
            loss += torch.mean((fake_output[i]-1)**2) 
        return loss

    def gen_loss2(self, fake, fake_lr):
        fake_output = self.forward(fake, fake_lr)
        loss = 0
        for i in range(len(fake_output)):
            loss += torch.mean((fake_output[i]-1)**2) 
        return loss


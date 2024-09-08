import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import numpy as np
import os
import cv2
from PIL import Image
from copy import deepcopy
from measure import measure_single_path_wo_lpips

#LU keep
# x = np.pad(x, ((0,1), (0,1)), mode='reflect')
#RU keep
# x = np.pad(x, ((0,1), (1,0)), mode='reflect')
#LD keep
# x = np.pad(x, ((1,0), (0,1)), mode='reflect')
#RD
# x = np.pad(x, ((1,0), (1,0)), mode='reflect')
def crop_cpu(img,crop_sz,step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h=x + crop_sz
    w=y + crop_sz
    return lr_list,num_h, num_w,h,w

def combine(sr_list,num_h, num_w,h,w,patch_size,step, scale):
    index=0
    sr_img = np.zeros((h*scale, w*scale, 3), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i*step*scale:i*step*scale+patch_size*scale, j*step*scale:j*step*scale+patch_size*scale, :]+=sr_list[index]
            index+=1
    sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step*scale:j*step*scale+(patch_size-step)*scale,:]/=2 

    for i in range(1,num_h):
        sr_img[i*step*scale:i*step*scale+(patch_size-step)*scale,:,:]/=2
    return sr_img

def reduce_spatical_dimension(lr_padding, H, W, C):
    out = np.zeros((H, W, C))
    for i in range(H):
        for j in range(W):
            for c in range(C):
                block = lr_padding[i:i+RF_SIZE, j:j+RF_SIZE, c]
                out[i, j , c] = np.mean(block)
    return out


def FourSimplexInterpBatchPixelGPU(LUT_tensor, Weight_tensor, patch, h, w, q, rot, upscale=4, sampling_interval=4, use_gpu=True):
    # LUT_tensor: [N, 83521, 4X4], Weight_tensor: [B, N, H, W], patch: [B, C, H, W]
    L = 2**(8-sampling_interval) + 1
    N, B, C, H, W = LUT_tensor.shape[0], patch.shape[0], patch.shape[1], h, w
    device = Weight_tensor.device
    # Extract MSBs
    img_a1 = patch[:, :, 0:0+h, 0:0+w]// q
    img_b1 = patch[:, :, 0:0+h, 1:1+w]// q
    img_c1 = patch[:, :, 1:1+h, 0:0+w]// q
    img_d1 = patch[:, :, 1:1+h, 1:1+w]// q

    img_a1 = img_a1.detach().cpu().numpy()
    img_b1 = img_b1.detach().cpu().numpy()
    img_c1 = img_c1.detach().cpu().numpy()
    img_d1 = img_d1.detach().cpu().numpy()

    # Extract LSBs
    fa_ = patch[:, :, 0:0+h, 0:0+w] % q
    fb_ = patch[:, :, 0:0+h, 1:1+w] % q
    fc_ = patch[:, :, 1:1+h, 0:0+w] % q
    fd_ = patch[:, :, 1:1+h, 1:1+w] % q

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    # Vertices (O in Eq3 and Table3 in the paper)
    #[N,B,C,H,W,4,4]
    p0000_list, p0001_list, p0010_list, p0011_list, p0100_list, p0101_list, p0110_list, p0111_list = torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device)
    p1000_list, p1001_list, p1010_list, p1011_list, p1100_list, p1101_list, p1110_list, p1111_list = torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device), torch.zeros([N,B,C,H,W,upscale,upscale]).to(device)
    
    for i in range(N):
        LUT = LUT_tensor[i]
        p0000_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0001_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0010_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0011_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0100_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0101_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0110_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0111_list[i] = LUT[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        
        p1000_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1001_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1010_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1011_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1100_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1101_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1110_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1111_list[i] = LUT[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))

    #adapt piiii with weight
    weight = Weight_tensor.transpose(0,1).unsqueeze(2).unsqueeze(-1).unsqueeze(-1) #[B,N,H,W] -> [N,B,1,H,W,1,1]
    weight = weight.repeat(1,1,C,1,1,upscale,upscale) #repeat [N,B,1,H,W,1,1] -> [N,B,C,H,W,4,4]
    # mul p * w :[N,B,C,H,W,4,4] * [N,B,C,H,W,4,4]
    p0000_list, p0001_list, p0010_list, p0011_list, p0100_list, p0101_list, p0110_list, p0111_list = p0000_list*weight, p0001_list*weight, p0010_list*weight, p0011_list*weight, p0100_list*weight, p0101_list*weight, p0110_list*weight, p0111_list*weight
    p1000_list, p1001_list, p1010_list, p1011_list, p1100_list, p1101_list, p1110_list, p1111_list = p1000_list*weight, p1001_list*weight, p1010_list*weight, p1011_list*weight, p1100_list*weight, p1101_list*weight, p1110_list*weight, p1111_list*weight
    #sum [B,C,H,W,4,4]
    p0000_list, p0001_list, p0010_list, p0011_list, p0100_list, p0101_list, p0110_list, p0111_list = torch.sum(p0000_list, dim=0), torch.sum(p0001_list, dim=0), torch.sum(p0010_list, dim=0), torch.sum(p0011_list, dim=0), torch.sum(p0100_list, dim=0), torch.sum(p0101_list, dim=0), torch.sum(p0110_list, dim=0), torch.sum(p0111_list, dim=0)
    p1000_list, p1001_list, p1010_list, p1011_list, p1100_list, p1101_list, p1110_list, p1111_list = torch.sum(p1000_list, dim=0), torch.sum(p1001_list, dim=0), torch.sum(p1010_list, dim=0), torch.sum(p1011_list, dim=0), torch.sum(p1100_list, dim=0), torch.sum(p1101_list, dim=0), torch.sum(p1110_list, dim=0), torch.sum(p1111_list, dim=0)

    # Output image holder
    #[B,C,H,W,4,4]
    out = torch.zeros([img_a1.shape[0],img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale], dtype=Weight_tensor.dtype).to(device)
    # Naive pixelwise output value interpolation
    # It would be faster implemented with a parallel operation
    
    #get interpolation matrix
    fa_interpolation = fa_.unsqueeze(-1).unsqueeze(-1)
    fb_interpolation = fb_.unsqueeze(-1).unsqueeze(-1)
    fc_interpolation = fc_.unsqueeze(-1).unsqueeze(-1)
    fd_interpolation = fd_.unsqueeze(-1).unsqueeze(-1)
    #abcd,abdc,adbc,dabc,acbd,acdb,adcb,dacb,cabd,cadb,cdab,dcab
    #bacd,badc,bdac,dbac,bcad,bcda,bdca,dbca,cbad,cbda,cdba,dcba
    matrix1 = (q-fa_interpolation) * p0000_list + (fa_interpolation-fb_interpolation) * p1000_list + (fb_interpolation-fc_interpolation) * p1100_list + (fc_interpolation-fd_interpolation) * p1110_list + (fd_interpolation) * p1111_list
    matrix2 = (q-fa_interpolation) * p0000_list + (fa_interpolation-fb_interpolation) * p1000_list + (fb_interpolation-fd_interpolation) * p1100_list + (fd_interpolation-fc_interpolation) * p1101_list + (fc_interpolation) * p1111_list
    matrix3 = (q-fa_interpolation) * p0000_list + (fa_interpolation-fd_interpolation) * p1000_list + (fd_interpolation-fb_interpolation) * p1001_list + (fb_interpolation-fc_interpolation) * p1101_list + (fc_interpolation) * p1111_list
    matrix4 = (q-fd_interpolation) * p0000_list + (fd_interpolation-fa_interpolation) * p0001_list + (fa_interpolation-fb_interpolation) * p1001_list + (fb_interpolation-fc_interpolation) * p1101_list + (fc_interpolation) * p1111_list
    matrix5 = (q-fa_interpolation) * p0000_list + (fa_interpolation-fc_interpolation) * p1000_list + (fc_interpolation-fb_interpolation) * p1010_list + (fb_interpolation-fd_interpolation) * p1110_list + (fd_interpolation) * p1111_list
    matrix6 = (q-fa_interpolation) * p0000_list + (fa_interpolation-fc_interpolation) * p1000_list + (fc_interpolation-fd_interpolation) * p1010_list + (fd_interpolation-fb_interpolation) * p1011_list + (fb_interpolation) * p1111_list
    matrix7 = (q-fa_interpolation) * p0000_list + (fa_interpolation-fd_interpolation) * p1000_list + (fd_interpolation-fc_interpolation) * p1001_list + (fc_interpolation-fb_interpolation) * p1011_list + (fb_interpolation) * p1111_list
    matrix8 = (q-fd_interpolation) * p0000_list + (fd_interpolation-fa_interpolation) * p0001_list + (fa_interpolation-fc_interpolation) * p1001_list + (fc_interpolation-fb_interpolation) * p1011_list + (fb_interpolation) * p1111_list
    matrix9 = (q-fc_interpolation) * p0000_list + (fc_interpolation-fa_interpolation) * p0010_list + (fa_interpolation-fb_interpolation) * p1010_list + (fb_interpolation-fd_interpolation) * p1110_list + (fd_interpolation) * p1111_list
    matrix10= (q-fc_interpolation) * p0000_list + (fc_interpolation-fa_interpolation) * p0010_list + (fa_interpolation-fd_interpolation) * p1010_list + (fd_interpolation-fb_interpolation) * p1011_list + (fb_interpolation) * p1111_list
    matrix11= (q-fc_interpolation) * p0000_list + (fc_interpolation-fd_interpolation) * p0010_list + (fd_interpolation-fa_interpolation) * p0011_list + (fa_interpolation-fb_interpolation) * p1011_list + (fb_interpolation) * p1111_list
    matrix12= (q-fd_interpolation) * p0000_list + (fd_interpolation-fc_interpolation) * p0001_list + (fc_interpolation-fa_interpolation) * p0011_list + (fa_interpolation-fb_interpolation) * p1011_list + (fb_interpolation) * p1111_list
    matrix13= (q-fb_interpolation) * p0000_list + (fb_interpolation-fa_interpolation) * p0100_list + (fa_interpolation-fc_interpolation) * p1100_list + (fc_interpolation-fd_interpolation) * p1110_list + (fd_interpolation) * p1111_list
    matrix14= (q-fb_interpolation) * p0000_list + (fb_interpolation-fa_interpolation) * p0100_list + (fa_interpolation-fd_interpolation) * p1100_list + (fd_interpolation-fc_interpolation) * p1101_list + (fc_interpolation) * p1111_list
    matrix15= (q-fb_interpolation) * p0000_list + (fb_interpolation-fd_interpolation) * p0100_list + (fd_interpolation-fa_interpolation) * p0101_list + (fa_interpolation-fc_interpolation) * p1101_list + (fc_interpolation) * p1111_list
    matrix16= (q-fd_interpolation) * p0000_list + (fd_interpolation-fb_interpolation) * p0001_list + (fb_interpolation-fa_interpolation) * p0101_list + (fa_interpolation-fc_interpolation) * p1101_list + (fc_interpolation) * p1111_list
    matrix17= (q-fb_interpolation) * p0000_list + (fb_interpolation-fc_interpolation) * p0100_list + (fc_interpolation-fa_interpolation) * p0110_list + (fa_interpolation-fd_interpolation) * p1110_list + (fd_interpolation) * p1111_list
    matrix18= (q-fb_interpolation) * p0000_list + (fb_interpolation-fc_interpolation) * p0100_list + (fc_interpolation-fd_interpolation) * p0110_list + (fd_interpolation-fa_interpolation) * p0111_list + (fa_interpolation) * p1111_list
    matrix19= (q-fb_interpolation) * p0000_list + (fb_interpolation-fd_interpolation) * p0100_list + (fd_interpolation-fc_interpolation) * p0101_list + (fc_interpolation-fa_interpolation) * p0111_list + (fa_interpolation) * p1111_list
    matrix20= (q-fd_interpolation) * p0000_list + (fd_interpolation-fb_interpolation) * p0001_list + (fb_interpolation-fc_interpolation) * p0101_list + (fc_interpolation-fa_interpolation) * p0111_list + (fa_interpolation) * p1111_list
    matrix21= (q-fc_interpolation) * p0000_list + (fc_interpolation-fb_interpolation) * p0010_list + (fb_interpolation-fa_interpolation) * p0110_list + (fa_interpolation-fd_interpolation) * p1110_list + (fd_interpolation) * p1111_list
    matrix22= (q-fc_interpolation) * p0000_list + (fc_interpolation-fb_interpolation) * p0010_list + (fb_interpolation-fd_interpolation) * p0110_list + (fd_interpolation-fa_interpolation) * p0111_list + (fa_interpolation) * p1111_list
    matrix23= (q-fc_interpolation) * p0000_list + (fc_interpolation-fd_interpolation) * p0010_list + (fd_interpolation-fb_interpolation) * p0011_list + (fb_interpolation-fa_interpolation) * p0111_list + (fa_interpolation) * p1111_list
    matrix24= (q-fd_interpolation) * p0000_list + (fd_interpolation-fc_interpolation) * p0001_list + (fc_interpolation-fb_interpolation) * p0011_list + (fb_interpolation-fa_interpolation) * p0111_list + (fa_interpolation) * p1111_list

    #Conditional judgment   
    #[b,c,h,w]
    if_a_b = (fa_ - fb_).gt(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_a_c = (fa_ - fc_).gt(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_a_d = (fa_ - fd_).gt(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_b_c = (fb_ - fc_).gt(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_b_d = (fb_ - fd_).gt(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_c_d = (fc_ - fd_).gt(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_b_a_ = (fb_ - fa_).ge(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_c_a_ = (fc_ - fa_).ge(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_d_a_ = (fd_ - fa_).ge(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_c_b_ = (fc_ - fb_).ge(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_d_b_ = (fd_ - fb_).ge(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()
    if_d_c_ = (fd_ - fc_).ge(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)#.detach().cpu().numpy()

    # abcd
    # c=(if_a_b*if_b_c*if_c_d).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,upscale,upscale)
    out = out + (if_a_b*if_b_c*if_c_d) * matrix1
    # m = out
    # abdc
    out = out + (if_a_b*if_b_c*if_d_c_*if_b_d) * matrix2
    # adbc
    out = out + (if_a_b*if_b_c*if_d_c_*if_d_b_*if_a_d) * matrix3
    # dabc
    out = out + (if_a_b*if_b_c*if_d_c_*if_d_b_*if_d_a_) * matrix4
    # acbd
    out = out + (if_a_b*if_c_b_*if_a_c*if_b_d) * matrix5
    # acdb
    out = out + (if_a_b*if_c_b_*if_a_c*if_d_b_*if_c_d) * matrix6
    # adcb
    out = out + (if_a_b*if_c_b_*if_a_c*if_d_b_*if_d_c_*if_a_d) * matrix7
    # dacb
    out = out + (if_a_b*if_c_b_*if_a_c*if_d_b_*if_d_c_*if_d_a_) * matrix8
    # cabd
    out = out + (if_a_b*if_c_b_*if_c_a_*if_b_d) * matrix9
    # cadb
    out = out + (if_a_b*if_c_b_*if_c_a_*if_d_b_*if_c_d) * matrix10
    # cdab
    out = out + (if_a_b*if_c_b_*if_c_a_*if_d_b_*if_d_c_*if_a_d) * matrix11
    # dcab
    out = out + (if_a_b*if_c_b_*if_c_a_*if_d_b_*if_d_c_*if_d_a_) * matrix12
    # bacd
    out = out + (if_b_a_*if_a_c*if_c_d) * matrix13
    # badc
    out = out + (if_b_a_*if_a_c*if_d_c_*if_a_d) * matrix14
    # bdac
    out = out + (if_b_a_*if_a_c*if_d_c_*if_d_a_*if_b_d) * matrix15
    # dbac
    out = out + (if_b_a_*if_a_c*if_d_c_*if_d_a_*if_d_b_) * matrix16
    # bcad
    out = out + (if_b_a_*if_c_a_*if_b_c*if_a_d) * matrix17
    # bcda
    out = out + (if_b_a_*if_c_a_*if_b_c*if_d_a_*if_c_d) * matrix18
    # bdca
    out = out + (if_b_a_*if_c_a_*if_b_c*if_d_a_*if_d_c_*if_b_d) * matrix19
    # dbca
    out = out + (if_b_a_*if_c_a_*if_b_c*if_d_a_*if_d_c_*if_d_b_) * matrix20
    # cbad
    out = out + (if_b_a_*if_c_a_*if_c_b_*if_a_d) * matrix21
    # cbda
    out = out + (if_b_a_*if_c_a_*if_c_b_*if_d_a_*if_b_d) * matrix22
    # cdba
    out = out + (if_b_a_*if_c_a_*if_c_b_*if_d_a_*if_d_b_*if_c_d) * matrix23
    # dcba
    out = out + (if_b_a_*if_c_a_*if_c_b_*if_d_a_*if_d_b_*if_d_c_) * matrix24
        
    # (b, 3, 128, 128, 4, 4)
    # (0, 1,3, 2,4)
    out = out.permute(0, 1, 2,4, 3,5).reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]*upscale, img_a1.shape[3]*upscale))
    out = out / q
    return out

def SR_4DLUT_patch(lr_tensor, LUT_list, Weight_list, scale, sampling_interval):
    # Sampling interval for input
    q = 2**sampling_interval

    b_, c_, h_, w_ = lr_tensor.shape[0], lr_tensor.shape[1], lr_tensor.shape[2], lr_tensor.shape[3]
    # Paddding
    pad = nn.ReflectionPad2d(padding=(0,1,0,1))
    lr_pad_tensor = pad(lr_tensor)
    out_r0 = FourSimplexInterpBatchPixelGPU(LUT_list, Weight_list, lr_pad_tensor, h_, w_, q, 0, scale, sampling_interval)
    return out_r0


def _np2Tensor(l, rgb_range=255):
    def _single_np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_single_np2Tensor(_l) for _l in l]

if __name__ == '__main__':
    SCALE = 4
    RF_SIZE = 8
    STEP = 6
    SAMPLING_INTERVAL = 4        # N bit uniform sampling
    device = torch.device('cuda')
    LUT_LU_PATH = "/home/v-gyin/github/RCAN_LUT/RCAN_x4_4D_LU_4bit_uint8.npy"
    LUT_SR_PATH = '/home/v-gyin/github/SR-LUT/3_Test_using_LUT/Model_S_x4_4bit_uint8.npy'
    # Load LUT

    LUT_LU = np.load(LUT_LU_PATH).astype(np.float32).reshape(-1, SCALE*SCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
    LUT_SR = np.load(LUT_SR_PATH).astype(np.float32).reshape(-1, SCALE*SCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
    lr_dir = '/home/v-gyin/github/RCAN_LUT/Set5_BIx{}/'.format(SCALE)
    hr_dir = '/home/v-gyin/github/RCAN_LUT/Set5_HR/'
    lr_path_list = glob.glob(os.path.join(lr_dir, '*.png'))
    lr_path_list.sort()
    hr_path_list = glob.glob(os.path.join(hr_dir, '*.png'))
    hr_path_list.sort()

    psnr_list, ssim_list = [], []

    save_dir = '/home/v-gyin/github/RCAN_LUT/OutputX{}_fusion_LUT_{}/'.format(SCALE, RF_SIZE)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for lr_path, hr_path in zip(lr_path_list, hr_path_list):
        name = os.path.basename(lr_path) 
        sr_save_path = os.path.join(save_dir, name)

        lr_origin = cv2.imread(lr_path)
        lr_RGB = cv2.cvtColor(lr_origin, cv2.COLOR_BGR2RGB)

        hr_origin = cv2.imread(hr_path)
        # hr_RGB = cv2.cvtColor(hr_origin, cv2.COLOR_BGR2RGB)
        h, w, c = lr_RGB.shape[0], lr_RGB.shape[1], lr_RGB.shape[2]

        lr = _np2Tensor([lr_RGB])[0].unsqueeze(0)
        weight = torch.rand((lr.shape[0],2,lr.shape[2],lr.shape[3]))
        # [17^4,4,4]
        LUT_tensor = torch.from_numpy(np.array([LUT_LU, LUT_SR])).to(device)

        #[32,3,32,32]
        sr_patch1 = SR_4DLUT_patch(lr, LUT_tensor, weight, SCALE, SAMPLING_INTERVAL, inner_mul=False)
        sr_patch2 = SR_4DLUT_patch(lr, LUT_tensor, weight, SCALE, SAMPLING_INTERVAL, inner_mul=True)
        diff = sr_patch1 - sr_patch2
        c= 0
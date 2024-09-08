import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import glob
import numpy as np
import os
import cv2
from PIL import Image
import random

seed = random.randint(0, 2 ** 32)
random.seed(seed)

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

class LUTWeightPredDataSet(Dataset):
    def __init__(self, args, is_train=True):
        super(LUTWeightPredDataSet, self).__init__()
        self.args = args
        self.is_train = is_train
        self.scale_factor = int(args.scale_factor)
        self.patch_size_for_weight_prediction = self.args.patch_size_for_weight_prediction
        self.patch_size_for_LUT = self.args.patch_size_for_LUT

        if self.is_train:
            self.hr_file_path = self.args.train_hr_file_path
            self.lr_file_path = self.args.train_lr_file_path
        else:
            self.hr_file_path = self.args.test_hr_file_path
            self.lr_file_path = self.args.test_lr_file_path
        
        self.lr_path_list = glob.glob(os.path.join(self.lr_file_path, '*.png'))
        self.lr_path_list.sort()
        self.hr_path_list = glob.glob(os.path.join(self.hr_file_path, '*.png'))
        self.hr_path_list.sort()

        self.dataset_len = self.__len__()

    def __getitem__(self, idx):
        # onle lr ,hr  
        lr, hr, hr_filename, lr_filename = self._load_file(idx)
        if self.is_train:
            lr_weight_patches, lr_center_patches, hr_weight_patches, hr_center_patches, lr_filenames, hr_filenames = [], [], [], [], [], []
            bias1, bias2 = int((self.patch_size_for_weight_prediction - self.patch_size_for_LUT) // 2), int((self.patch_size_for_weight_prediction + self.patch_size_for_LUT) // 2)

            for i in range(self.args.batch_size_inner_patch):
                lr_weight_patch, hr_weight_patch = self._get_pair_patch(lr, hr, lr_size=int(self.patch_size_for_weight_prediction))
                lr_weight_patch, hr_weight_patch = self._set_img_channel([lr_weight_patch, hr_weight_patch], img_mode="RGB")
                lr_weight_patch, hr_weight_patch = self._np2Tensor([lr_weight_patch, hr_weight_patch], rgb_range=self.args.rgb_range)
                
                lr_weight_patches.append(lr_weight_patch.unsqueeze(0))
                hr_weight_patches.append(hr_weight_patch.unsqueeze(0))
                lr_filenames.append(lr_filename)
                hr_filenames.append(hr_filename)
            
            lr_weight_patches = torch.cat(lr_weight_patches, 0)
            hr_weight_patches = torch.cat(hr_weight_patches, 0)
        
            lr_center_patches, hr_center_patches = lr_weight_patches[:, :, bias1:bias2, bias1:bias2], hr_weight_patches[:, :, self.scale_factor*bias1:self.scale_factor*bias2, self.scale_factor*bias1:self.scale_factor*bias2]
            return lr_weight_patches, lr_center_patches, hr_weight_patches, hr_center_patches, lr_filenames, hr_filenames
        else:
            lr_RGB, hr_RGB = self._set_img_channel([lr, hr], img_mode="RGB")
            # lr_crop_RGB, hr_crop_RGB = lr_RGB[bias1:-bias1], hr_RGB[bias1*self.scale_factor:-bias1*self.scale_factor] 
            return lr_RGB, hr_RGB

    def __len__(self):
        if self.is_train:
            return len(self.hr_path_list)  # * self.repeat
        else:
            return len(self.lr_path_list)
    
    def _get_index(self, idx):
        if self.is_train:
            return idx % len(self.hr_path_list)
        else:
            return idx
    
    def _load_file(self, idx):
        idx = self._get_index(idx)

        hr_path = self.hr_path_list[idx]
        hr = cv2.imread(hr_path)
        hr_filename = os.path.basename(hr_path) 

        lr_path = self.lr_path_list[idx]
        lr = cv2.imread(lr_path)
        lr_filename = os.path.basename(lr_path) 

        return lr, hr, hr_filename, lr_filename
    
    def _get_pair_patch(self, lr, hr, lr_size=8):
        ih, iw = lr.shape[:2]
        # print(ih,iw)

        tp = int(lr_size * self.scale_factor)
        ip = int (lr_size)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        tx, ty = self.scale_factor * ix, self.scale_factor * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        hr_patch = hr[ty:ty + tp, tx:tx + tp, :]

        return lr_patch, hr_patch
    
    def _set_img_channel(self, l, img_mode="RGB"):
        def _set_single_img_channel(img, img_mode):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            c = img.shape[2]
            if img_mode == "YCbCr" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif img_mode == "YCbCr" and c == 1:
                img = np.concatenate([img] * 3, 2)
            if img_mode == "RGB" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        return [_set_single_img_channel(_l, img_mode) for _l in l]

    def _np2Tensor(self, l, rgb_range):
        def _single_np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)

            return tensor

        return [_single_np2Tensor(_l) for _l in l]


class LUTPixelDataSet(Dataset):
    def __init__(self, args, is_train=True):
        super(LUTPixelDataSet, self).__init__()
        self.args = args
        self.is_train = is_train
        self.scale_factor = int(args.scale_factor)
        self.patch_pixel_size = self.args.patch_pixel_size

        if self.is_train:
            self.hr_file_path = self.args.train_hr_file_path
            self.lr_file_path = self.args.train_lr_file_path
        else:
            self.hr_file_path = self.args.test_hr_file_path
            self.lr_file_path = self.args.test_lr_file_path
        
        self.lr_path_list = glob.glob(os.path.join(self.lr_file_path, '*.png'))
        if self.is_train:
            self.lr_path_list += glob.glob(os.path.join('/home/v-gyin/dataset/Set5/Set5_BIx4/', '*.png'))
        self.lr_path_list.sort()
        self.hr_path_list = glob.glob(os.path.join(self.hr_file_path, '*.png'))
        if self.is_train:
            self.hr_path_list += glob.glob(os.path.join('/home/v-gyin/dataset/Set5/Set5_HR/', '*.png'))
        self.hr_path_list.sort()

        self.dataset_len = self.__len__()

    def __getitem__(self, idx):
        # onle lr ,hr  
        lr, hr, hr_filename, lr_filename = self._load_file(idx)
        if self.is_train:
            lr_patch, hr_patch = self._get_pair_patch(lr, hr, lr_size=int(self.patch_pixel_size))
            lr_patch_up =  cv2.resize(lr_patch, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
            lr_patch, lr_patch_up, hr_patch = self._set_img_channel([lr_patch, lr_patch_up, hr_patch], img_mode="RGB")
            lr_patch, lr_patch_up, hr_patch = self._np2Tensor([lr_patch, lr_patch_up, hr_patch], rgb_range=self.args.rgb_range)
                
            return lr_patch, lr_patch_up, hr_patch, lr_filename, hr_filename
        else:
            if self.args.self_ensemble_output:
                lr_up =  cv2.resize(lr, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
                lr1, lr_up1 = lr, lr_up
                lr2, lr_up2 = np.rot90(lr, 1), np.rot90(lr_up, 1)
                lr3, lr_up3 = np.rot90(lr, 2), np.rot90(lr_up, 2)
                lr4, lr_up4 = np.rot90(lr, 3), np.rot90(lr_up, 3)
                lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr = self._set_img_channel([lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr ], img_mode="RGB")
                lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr = self._np2Tensor([lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr], rgb_range=self.args.rgb_range)
                return lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr, lr_filename, hr_filename
            else:
                lr_up =  cv2.resize(lr, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
                lr, lr_up, hr = self._set_img_channel([lr, lr_up, hr], img_mode="RGB")
                lr, lr_up, hr = self._np2Tensor([lr, lr_up, hr], rgb_range=self.args.rgb_range)
                return lr, lr_up, hr, lr_filename, hr_filename

    def __len__(self):
        if self.is_train:
            return len(self.hr_path_list)  # * self.repeat
        else:
            return len(self.lr_path_list)
    
    def _get_index(self, idx):
        if self.is_train:
            return idx % len(self.hr_path_list)
        else:
            return idx
    
    def _load_file(self, idx):
        idx = self._get_index(idx)

        hr_path = self.hr_path_list[idx]
        hr = cv2.imread(hr_path)
        hr_filename = os.path.basename(hr_path) 

        lr_path = self.lr_path_list[idx]
        lr = cv2.imread(lr_path)
        lr_filename = os.path.basename(lr_path)

        if lr.shape[0] == 0 and lr.shape[1] == 0:
            raise RuntimeError('invalid {}'.format(lr_path))

        return lr, hr, hr_filename, lr_filename
    
    def _get_pair_patch(self, lr, hr, lr_size=8):
        ih, iw = lr.shape[:2]
        # print(ih,iw)

        tp = int(lr_size * self.scale_factor)
        ip = int (lr_size)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        tx, ty = self.scale_factor * ix, self.scale_factor * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        hr_patch = hr[ty:ty + tp, tx:tx + tp, :]

        return lr_patch, hr_patch
    
    def _set_img_channel(self, l, img_mode="RGB"):
        def _set_single_img_channel(img, img_mode):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            c = img.shape[2]
            if img_mode == "YCbCr" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif img_mode == "YCbCr" and c == 1:
                img = np.concatenate([img] * 3, 2)
            if img_mode == "RGB" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        return [_set_single_img_channel(_l, img_mode) for _l in l]

    def _np2Tensor(self, l, rgb_range):
        def _single_np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)

            return tensor

        return [_single_np2Tensor(_l) for _l in l]


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


def glob_second_dir(path, type='*.png', break_single=False):
    file_list = []
    for second_file in os.listdir(path):
        second_path = os.path.join(path, second_file)
        file_list += glob.glob(os.path.join(second_path, type))
        if break_single:
            return file_list
    return file_list


class LUTVSRDataSet(Dataset):
    def __init__(self, args, is_up=True, is_train=True):
        super(LUTVSRDataSet, self).__init__()
        self.args = args
        self.is_train = is_train
        self.is_up = is_up
        self.scale_factor = int(args.scale_factor)
        self.patch_size = self.args.patch_size
        self.frame_size = self.args.frame_size
        self.every_frame = self.args.every_frame
        self.self_ensemble_output = self.args.self_ensemble_output 

        if self.is_train:
            self.hr_file_path = self.args.train_hr_file_path
            self.lr_file_path = self.args.train_lr_file_path
            self.lr_mv_file_path = self.args.train_lr_mv_file_path
            
            break_single = False
            # self.lr_frames_list = self._glob_VSR_second_dir(self.lr_file_path, 2, '*.png', break_single)
            # self.lr_frames_list += self._glob_VSR_second_dir(self.args.val_lr_file_path, 2, '*.png', break_single)
            self.lr_frames_list = self._glob_VSR_second_dir(self.args.test_lr_file_path, 2, '*.png', break_single)
            self.lr_frames_list.sort()
            # self.hr_frames_list = self._glob_VSR_second_dir(self.hr_file_path, 2, '*.png', break_single)
            # self.hr_frames_list += self._glob_VSR_second_dir(self.args.val_hr_file_path, 2, '*.png', break_single)
            self.hr_frames_list = self._glob_VSR_second_dir(self.args.test_hr_file_path, 2, '*.png', break_single)
            self.hr_frames_list.sort()

        else:
            self.lr_file_path = self.args.test_lr_file_path
            self.lr_mv_file_path = self.args.test_lr_mv_file_path
            self.hr_file_path = self.args.test_hr_file_path
            break_single = True
            self.lr_frames_list = self._glob_VSR_second_dir(self.lr_file_path, 2, '*.png', break_single)
            self.lr_frames_list.sort()
            self.hr_frames_list = self._glob_VSR_second_dir(self.hr_file_path, 2, '*.png', break_single)
            self.hr_frames_list.sort()

        self.dataset_len = self.__len__()
    
    def _glob_VSR_second_dir(self, path, frames=None, type='*.png', break_single=False):
        file_frames_groups = []
        file_paths = os.listdir(path)
        file_paths.sort()
        for second_file in file_paths:
            second_path = os.path.join(path, second_file)
            file_video_list = glob.glob(os.path.join(second_path, type))
            file_video_list.sort()
            if frames == None:
                frames = len(file_video_list) - 1
            if len(file_video_list) < frames + 1:
                raise ValueError('invalid file_frames_groups len {} less than frames number {}'.format(len(file_video_list)))
            #多取一帧作为引导帧
            for i in range(len(file_video_list) - frames):
                file_frames_groups.append(file_video_list[i:i+frames+1])
            if break_single:
                break

        return file_frames_groups        

    def __getitem__(self, idx):
        # [np.img for _ in range(frames)]
        lr, lr_ref, lr_mv, hr, lr_path, hr_path = self._load_frames(idx)
        if self.is_up:
            lr_ref_up = cv2.resize(lr, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
            lr_mv_up = cv2.resize(lr_mv, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
        else:
            lr_ref_up, lr_mv_up = lr, lr_mv
        #train
        if self.is_train:
            lr, lr_ref_up, lr_mv_up, hr = self._get_quad_patch(lr, lr_ref_up, lr_mv_up, hr, lr_size=int(self.patch_size), is_up=self.is_up)
        if self.is_up:
            lr_up = cv2.resize(lr, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
        else:
            lr_up = lr
        lr, lr_up, lr_ref_up, lr_mv_up, hr = self._set_img_channel([lr, lr_up, lr_ref_up, lr_mv_up, hr], img_mode="RGB")
        lr, lr_up, lr_ref_up, lr_mv_up, hr = self._np2Tensor([lr, lr_up, lr_ref_up, lr_mv_up, hr], rgb_range=self.args.rgb_range)
            
        return lr, lr_up, lr_ref_up, lr_mv_up, hr, lr_path, hr_path 

    def __len__(self):
        if self.is_train:
            return len(self.hr_frames_list)  # * self.repeat
        else:
            return len(self.lr_frames_list)
    
    def _get_index(self, idx):
        if self.is_train:
            return idx % len(self.hr_frames_list)
        else:
            return idx
    
    def _load_frames(self, idx):
        idx = self._get_index(idx)
        hr_frames, lr_frames, lr_mv_frames = [], [], []
        hr_filenames, lr_filenames = [], []
        hr_frames_paths = self.hr_frames_list[idx]
        lr_frames_paths = self.lr_frames_list[idx]
        
        lr_path, lr_ref_path, hr_path = lr_frames_paths[1], lr_frames_paths[0], hr_frames_paths[1]
        lr_frame = cv2.imread(lr_frames_paths[1])
        lr_ref_frame = cv2.imread(lr_frames_paths[0])
        hr_frame = cv2.imread(hr_frames_paths[1])

        lr_mv_path = self.lr_mv_file_path + lr_path.split('/')[-2] + '/' + lr_path.split('/')[-1]
        if os.path.exists(lr_mv_path):
            lr_mv_frame = cv2.imread(lr_mv_path)
        else:
            lr_mv_frame = np.zeros(lr_frame.shape).astype(lr_frame.dtype)
            cv2.imwrite(lr_mv_path, lr_mv_frame)

        return lr_frame, lr_ref_frame, lr_mv_frame, hr_frame, lr_path, hr_path
    
    def _get_patch(self, lr, lr_size=8):
        ih, iw = lr.shape[:2]
        # print(ih,iw)

        ip = int (lr_size)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]

        return lr_patch

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
    
    def _get_triple_patch(self, lr, hr, ref_hr, lr_size=8):
        ih, iw = lr.shape[:2]
        # print(ih,iw)

        tp = int(lr_size * self.scale_factor)
        ip = int (lr_size)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        tx, ty = self.scale_factor * ix, self.scale_factor * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        hr_patch = hr[ty:ty + tp, tx:tx + tp, :]
        ref_hr_patch = ref_hr[ty:ty + tp, tx:tx + tp, :]

        return lr_patch, hr_patch, ref_hr_patch
    
    def _get_quad_patch(self, lr, lr_ref_up, lr_mv_up, hr, lr_size=8, is_up=True):
        if is_up:
            ih, iw = lr.shape[:2]
        # print(ih,iw)
        else:
            ih, iw = lr_mv_up.shape[:2]
        tp = int(lr_size * self.scale_factor)
        ip = int (lr_size)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        tx, ty = self.scale_factor * ix, self.scale_factor * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        if is_up:
            lr_ref_up_patch = lr_ref_up[ty:ty + tp, tx:tx + tp, :]
            lr_mv_up_patch = lr_mv_up[ty:ty + tp, tx:tx + tp, :]
        else:
            lr_ref_up_patch = lr_patch
            lr_mv_up_patch = lr_mv_up[iy:iy + ip, ix:ix + ip, :]
        hr_patch = hr[ty:ty + tp, tx:tx + tp, :]

        return lr_patch, lr_ref_up_patch, lr_mv_up_patch, hr_patch
    
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

if __name__ == '__main__':
    lr_file = '/home/v-gyin/dataset/100Kbps/Client/track3_test_png/'
    lr_mv_file = '/home/v-gyin/dataset/100Kbps/Client/track3_test_mv_png/'
    lr_file_paths = os.listdir(lr_file)
    lr_file_paths.sort()
    lr_mv_file_paths = os.listdir(lr_mv_file)
    lr_mv_file_paths.sort()
    for second_lr_file, second_lr_mv_file in zip(lr_file_paths, lr_mv_file_paths):
        second_lr_path = os.path.join(lr_file, second_lr_file)
        lr_file_video_list = glob.glob(os.path.join(second_lr_path, '*.png'))
        lr_file_video_list.sort()

        second_lr_mv_path = os.path.join(lr_mv_file, second_lr_mv_file)
        lr_mv_file_video_list = glob.glob(os.path.join(second_lr_mv_path, '*.png'))
        lr_mv_file_video_list.sort()

        for i in range(len(lr_mv_file_video_list)):
            mv_name = os.path.basename(lr_mv_file_video_list[i])
            lr_name = os.path.basename(lr_file_video_list[i+1])
            if mv_name != lr_name:
                c = 0
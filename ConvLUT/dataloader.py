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
    def __init__(self, args, is_train=True):
        super(LUTVSRDataSet, self).__init__()
        self.args = args
        self.is_train = is_train
        self.scale_factor = int(args.scale_factor)
        self.patch_size = self.args.patch_size
        self.frame_size = self.args.frame_size
        self.every_frame = self.args.every_frame
        self.self_ensemble_output = self.args.self_ensemble_output 

        if self.is_train:
            self.hr_file_path = self.args.train_hr_file_path
            self.lr_file_path = self.args.train_lr_file_path
            
            if self.is_train:
                break_single = False
                self.lr_frames_list = self._glob_VSR_second_dir(self.lr_file_path, self.frame_size, '*.png', break_single)
                # self.lr_frames_list += self._glob_VSR_second_dir(self.args.val_lr_file_path, self.frame_size, '*.png', break_single)
                # self.lr_frames_list += self._glob_VSR_second_dir(self.args.test_lr_file_path, self.frame_size, '*.png', break_single)
                self.lr_frames_list.sort()
                self.hr_frames_list = self._glob_VSR_second_dir(self.hr_file_path, self.frame_size, '*.png', break_single)
                # self.hr_frames_list += self._glob_VSR_second_dir(self.args.val_hr_file_path, self.frame_size, '*.png', break_single)
                # self.hr_frames_list += self._glob_VSR_second_dir(self.args.test_hr_file_path, self.frame_size, '*.png', break_single)
                self.hr_frames_list.sort()
        else:
            self.hr_file_path = self.args.test_hr_file_path
            self.lr_file_path = self.args.test_lr_file_path
            break_single = True
            self.lr_frames_list = self._glob_VSR_second_dir(self.lr_file_path, None, '*.png', break_single)
            self.lr_frames_list.sort()
            self.hr_frames_list = self._glob_VSR_second_dir(self.hr_file_path, None, '*.png', break_single)
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
        hr_origin_frames_np, lr_origin_frames_np, hr_filenames, lr_filenames = self._load_frames(idx)
        lr_frames, lr_up_frames, hr_frames = [], [], []
        first_ref_lr_frame_up = cv2.resize(lr_origin_frames_np[0], (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
        lr_frames_np = lr_origin_frames_np[1:]
        hr_frames_np = hr_origin_frames_np[1:]
        #train
        if self.is_train:
            for i in range(len(lr_frames_np)):
                lr, hr = lr_frames_np[i], hr_frames_np[i]
                if i == 0:
                    lr_patch, hr_patch, first_ref_lr_patch_up = self._get_triple_patch(lr, hr, first_ref_lr_frame_up, lr_size=int(self.patch_size))
                    lr_patch_up = cv2.resize(lr_patch, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
                    lr_patch, lr_patch_up, hr_patch, first_ref_lr_patch_up = self._set_img_channel([lr_patch, lr_patch_up, hr_patch, first_ref_lr_patch_up], img_mode="RGB")
                    lr_patch, lr_patch_up, hr_patch, first_ref_lr_patch_up = self._np2Tensor([lr_patch, lr_patch_up, hr_patch, first_ref_lr_patch_up], rgb_range=self.args.rgb_range)
                    lr_up_frames.append(first_ref_lr_patch_up.unsqueeze(0))
                else:
                    lr_patch, hr_patch = self._get_pair_patch(lr, hr, lr_size=int(self.patch_size))
                    lr_patch_up = cv2.resize(lr_patch, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
                    lr_patch, lr_patch_up, hr_patch = self._set_img_channel([lr_patch, lr_patch_up, hr_patch], img_mode="RGB")
                    lr_patch, lr_patch_up, hr_patch = self._np2Tensor([lr_patch, lr_patch_up, hr_patch], rgb_range=self.args.rgb_range)
                lr_frames.append(lr_patch.unsqueeze(0))
                lr_up_frames.append(lr_patch_up.unsqueeze(0))
                hr_frames.append(hr_patch.unsqueeze(0))
            lr_frames, lr_up_frames, hr_frames = torch.cat(lr_frames, 0), torch.cat(lr_up_frames, 0), torch.cat(hr_frames, 0)
            #[t,c,h,w], [t+1,c,h,w], [t,c,h,w]    
            return lr_frames, lr_up_frames, hr_frames, lr_filenames[1:], hr_filenames[1:]
        #test
        else:
            if self.self_ensemble_output:
                lr1_frames, lr_up1_frames, lr2_frames, lr_up2_frames, lr3_frames, lr_up3_frames, lr4_frames, lr_up4_frames, hr_frames = [], [], [], [], [], [], [], [], []
            else:
                lr_frames, lr_up_frames, hr_frames = [], [], []

            for i in range(len(lr_frames_np)):
                lr, hr = lr_frames_np[i], hr_frames_np[i]
                lr_up =  cv2.resize(lr, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
                if self.self_ensemble_output:
                    if i == 0:
                        first_ref_lr_frame_up1, first_ref_lr_frame_up2, first_ref_lr_frame_up3, first_ref_lr_frame_up4 = first_ref_lr_frame_up, np.rot90(first_ref_lr_frame_up, 1), np.rot90(first_ref_lr_frame_up, 2), np.rot90(first_ref_lr_frame_up, 3)
                        first_ref_lr_frame_up1, first_ref_lr_frame_up2, first_ref_lr_frame_up3, first_ref_lr_frame_up4 = self._set_img_channel([first_ref_lr_frame_up1, first_ref_lr_frame_up2, first_ref_lr_frame_up3, first_ref_lr_frame_up4], img_mode="RGB")
                        first_ref_lr_frame_up1, first_ref_lr_frame_up2, first_ref_lr_frame_up3, first_ref_lr_frame_up4 = self._np2Tensor([first_ref_lr_frame_up1, first_ref_lr_frame_up2, first_ref_lr_frame_up3, first_ref_lr_frame_up4], rgb_range=self.args.rgb_range)
                        lr_up1_frames.append(first_ref_lr_frame_up1.unsqueeze(0))
                        lr_up2_frames.append(first_ref_lr_frame_up2.unsqueeze(0))
                        lr_up3_frames.append(first_ref_lr_frame_up3.unsqueeze(0))
                        lr_up4_frames.append(first_ref_lr_frame_up4.unsqueeze(0))
                    lr1, lr_up1 = lr, lr_up
                    lr2, lr_up2 = np.rot90(lr, 1), np.rot90(lr_up, 1)
                    lr3, lr_up3 = np.rot90(lr, 2), np.rot90(lr_up, 2)
                    lr4, lr_up4 = np.rot90(lr, 3), np.rot90(lr_up, 3)
                    lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr = self._set_img_channel([lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr ], img_mode="RGB")
                    lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr = self._np2Tensor([lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr], rgb_range=self.args.rgb_range)
                    
                    lr1_frames.append(lr1.unsqueeze(0))
                    lr2_frames.append(lr2.unsqueeze(0))
                    lr3_frames.append(lr3.unsqueeze(0))
                    lr4_frames.append(lr4.unsqueeze(0))
                    lr_up1_frames.append(lr_up1.unsqueeze(0))
                    lr_up2_frames.append(lr_up2.unsqueeze(0))
                    lr_up3_frames.append(lr_up3.unsqueeze(0))
                    lr_up4_frames.append(lr_up4.unsqueeze(0))
                    hr_frames.append(hr.unsqueeze(0))
                    
                else:
                    if i == 0:
                        first_ref_lr_frame_up = self._set_img_channel([first_ref_lr_frame_up], img_mode="RGB")[0]
                        first_ref_lr_frame_up = self._np2Tensor([first_ref_lr_frame_up], rgb_range=self.args.rgb_range)[0]
                        lr_up_frames.append(first_ref_lr_frame_up.unsqueeze(0))
                    lr_up =  cv2.resize(lr, (0,0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
                    lr, lr_up, hr = self._set_img_channel([lr, lr_up, hr], img_mode="RGB")
                    lr, lr_up, hr = self._np2Tensor([lr, lr_up, hr], rgb_range=self.args.rgb_range)
                    lr_frames.append(lr.unsqueeze(0))
                    lr_up_frames.append(lr_up.unsqueeze(0))
                    hr_frames.append(hr.unsqueeze(0))
                
            if self.self_ensemble_output:
                lr1_frames, lr_up1_frames, lr2_frames, lr_up2_frames, lr3_frames, lr_up3_frames, lr4_frames, lr_up4_frames, hr_frames = torch.cat(lr1_frames,0), torch.cat(lr_up1_frames,0), torch.cat(lr2_frames,0), torch.cat(lr_up2_frames,0), torch.cat(lr3_frames,0), torch.cat(lr_up3_frames,0), torch.cat(lr4_frames,0), torch.cat(lr_up4_frames,0), torch.cat(hr_frames,0)
                return lr1_frames, lr_up1_frames, lr2_frames, lr_up2_frames, lr3_frames, lr_up3_frames, lr4_frames, lr_up4_frames, hr_frames, lr_filenames[1:], hr_filenames[1:]
            else:
                lr_frames, lr_up_frames, hr_frames = torch.cat(lr_frames,0), torch.cat(lr_up_frames,0), torch.cat(hr_frames,0)
                return lr_frames, lr_up_frames, hr_frames, lr_filenames[1:], hr_filenames[1:]        

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
        hr_frames, lr_frames = [], []
        hr_filenames, lr_filenames = [], []
        hr_frames_paths = self.hr_frames_list[idx]
        lr_frames_paths = self.lr_frames_list[idx]
        for hr_path, lr_path in zip(hr_frames_paths, lr_frames_paths):
            hr = cv2.imread(hr_path)
            lr = cv2.imread(lr_path)
            hr_filename = os.path.basename(hr_path) 
            lr_filename = os.path.basename(lr_path)
            hr_frames.append(hr)
            lr_frames.append(lr)
            hr_filenames.append(hr_filename)
            lr_filenames.append(lr_filename)

        return hr_frames, lr_frames, hr_frames_paths, lr_frames_paths
    
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


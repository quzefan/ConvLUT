import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2
import argparse

from natsort import natsort
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips


class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        if not all([s1 == s2 for s1, s2 in zip(imgA.shape, imgB.shape)]):
            raise RuntimeError("Image sizes not the same.")
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]
    
    def measure_wo_lpips(self, imgA, imgB):
        if not all([s1 == s2 for s1, s2 in zip(imgA.shape, imgB.shape)]):
            raise RuntimeError("Image sizes not the same.")
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        score, diff = compare_ssim(imgA, imgB, full=True, multichannel=True)
        return score

    def psnr(self, imgA, imgB):
        psnr = compare_psnr(imgA, imgB)
        return psnr


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    # files = glob.glob(r"{}/*.png".format(wildcard))
    # files.sort()
    # return files
    return natsort.natsorted(glob.glob(r"{}/*.png".format(wildcard), recursive=True))


def imread(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips):
    return f'{psnr:0.2f}, {ssim:0.3f}, {lpips:0.3f}'

def measure_dirs(dirA, dirB, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    # paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    # paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    paths_A = fiFindByWildcard(dirA)
    paths_B = fiFindByWildcard(dirB)


    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        A = imread(pathA)
        B = imread(pathB)
        # A = cv2.resize(A, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        # crop_border = 6+4
        # [A, B] = [img[crop_border:-crop_border, crop_border:-crop_border, :] for img in [A, B]]
        result['psnr'], result['ssim'], result['lpips'] = measure.measure(A, B)
        d = time.time() - t
        vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        results.append(result)

    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])

    vprint(f"Final Result: {format_result(psnr, ssim, lpips)}, {time.time() - t_init:0.1f}s")

def measure_single_path_wo_lpips(path_A, path_B, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    # paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    # paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))


    measure = Measure(use_gpu=use_gpu)

    t = time.time()
    A = imread(path_A)
    B = imread(path_B)
    psnr, ssim = measure.measure_wo_lpips(A, B)
    d = time.time() - t

    # vprint(f"Final Result: {format_result(psnr, ssim)}, {time.time() - t_init:0.1f}s")

    return psnr, ssim

def measure_single_img_wo_lpips(A, B, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    # paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    # paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))


    measure = Measure(use_gpu=use_gpu)

    t = time.time()
    psnr, ssim = measure.measure_wo_lpips(A, B)
    d = time.time() - t

    vprint(f"Final Result: {format_result(psnr, ssim)}, {time.time() - t_init:0.1f}s")

    return psnr, ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dirA', default='/opt/tiger/lab/wangwei/DAUSR/Domain_Cross/Domain_Cross_rgb_lsgan_pixx10_pixy10_cycle10_ganx1_identity1_vgg01_patch96_batch12_pretrain10wan_resblock5_rrdb10_msD_SR_CVPR2020_bicubic_catnoise1_v91/CVPR2020_track1_150000/sr', type=str)
    parser.add_argument('-dirB', default='/opt/tiger/lab/wangwei/DAUSR/valid-gt-clean', type=str)
    parser.add_argument('-type', default='png')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    type = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, use_gpu=use_gpu, verbose=True)

import torch
import quadrilinear4d
import torch.nn as nn
import numpy as np
import glob
import os
import cv2
from measure import measure_single_path_wo_lpips

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Weighted4DLUTInter(nn.Module):
    def __init__(self, dim=17, scale_factor=4, LUT_file=None, tri_index_file=None):
        super(Weighted4DLUTInter, self).__init__()
        if LUT_file == None:
            self.LUTs = torch.ones(1, dim,dim,dim,dim,scale_factor,scale_factor, dtype=torch.float)
        else:
            #load all npy from LUT_file
            LUT_list = glob.glob(os.path.join(LUT_file, '*.npy'))
            self.LUT_tables = []
            for LUT_path in LUT_list:
                LUT_table = np.load(LUT_path).astype(np.float32).reshape(dim,dim,dim,dim,scale_factor,scale_factor)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
                self.LUT_tables.append(LUT_table)
            self.LUT_tables_num = len(self.LUT_tables)
            if self.LUT_tables_num == 0:
                raise RuntimeError('LUT table load error!! Empty')
            self.LUTs = torch.from_numpy(np.array(self.LUT_tables))
        self.LUTs = nn.Parameter(torch.tensor(self.LUTs))
        
        if tri_index_file == None:
            raise ValueError("invalid tri_index_file path!!")
        tri_index = np.load(tri_index_file).astype(np.float32)
        self.tri_index = torch.from_numpy(np.array(tri_index))
        self.tri_index = nn.Parameter(torch.tensor(self.tri_index))
        
        self.QuadrilinearInterpolation = QuadrilinearInterpolation()

    def forward(self, weight, x, scale_factor=4):
        _, output = self.QuadrilinearInterpolation(self.LUTs, self.tri_index, weight, x, scale_factor)

        return output

class QuadrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, tri_index, weight, x, scale_factor):
        # x = x.contiguous()
        # y = x.size()
        output = torch.zeros([x.shape[0],x.shape[1], x.shape[2], x.shape[3], scale_factor, scale_factor], dtype=x.dtype).to(x.device)
        lut_num = lut.size()[0]
        dim = lut.size()[1]
        shift = scale_factor
        binsize = 16
        H = x.size(2)
        W = x.size(3)
        batch = x.size(0)
        
        pad = nn.ReflectionPad2d(padding=(0,1,0,1))
        x_pad = pad(x)
        x1 = x_pad[:, :, 0:0+H, 0:0+W].contiguous()
        x2 = x_pad[:, :, 0:0+H, 1:1+W].contiguous()
        x3 = x_pad[:, :, 1:1+H, 0:0+W].contiguous()
        x4 = x_pad[:, :, 1:1+H, 1:1+W].contiguous()
        
        assert 1 == quadrilinear4d.forward(lut,
                                      tri_index,
                                      weight, 
                                      x1,
                                      x2,
                                      x3,
                                      x4,
                                      output,
                                      lut_num,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)

        int_package = torch.IntTensor([lut_num, dim, shift, binsize, W, H, batch])
        variables = [lut, tri_index, weight, x, x1, x2, x3, x4, int_package]
        
        ctx.save_for_backward(*variables)
        
        output = output.permute(0, 1, 2, 4, 3,5).reshape((x.shape[0], x.shape[1], x.shape[2]*scale_factor, x.shape[3]*scale_factor))
        return weight, output
    
    @staticmethod
    def backward(ctx, weight_grad, output_grad):
        
        lut, tri_index, weight, x, x1, x2, x3, x4, int_package = ctx.saved_variables
        lut_num, dim, shift, binsize, W, H, batch = int_package
        lut_num, dim, shift, binsize, W, H, batch = int(lut_num), int(dim), int(shift), int(binsize), int(W), int(H), int(batch)
        

        assert 1 == quadrilinear4d.backward(lut,
                                      tri_index,
                                      weight_grad, 
                                      x1,
                                      x2,
                                      x3,
                                      x4,
                                      output_grad,
                                      lut_num,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)
        return None, None, weight_grad, None, None


class QuadrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(QuadrilinearInterpolation, self).__init__()

    def forward(self, lut, tri_index, weight, x, scale_factor):
        return QuadrilinearInterpolationFunction.apply(lut, tri_index, weight, x, scale_factor)

def _np2Tensor(l, rgb_range=255):
    def _single_np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_single_np2Tensor(_l) for _l in l]

# if __name__ == '__main__':
#     LUT_file = '/home/v-gyin/github/VSR_LUT_hidden/LUT/'
#     tri_index_file = '/home/v-gyin/github/VSR_LUT_hidden/triangular_index.npy'
#     LUT = Weighted4DLUTInter(LUT_file=LUT_file, tri_index_file=tri_index_file).cuda()

#     lr_dir = '/home/v-gyin/github/RCAN_LUT/Set5_BIx4/'
#     hr_dir = '/home/v-gyin/github/RCAN_LUT/Set5_HR/'
#     lr_path_list = glob.glob(os.path.join(lr_dir, '*.png'))
#     lr_path_list.sort()
#     hr_path_list = glob.glob(os.path.join(hr_dir, '*.png'))
#     hr_path_list.sort()

#     psnr_list, ssim_list = [], []

#     save_dir = '/home/v-gyin/github/VSR_LUT_hidden/CUDA_test/'

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for lr_path, hr_path in zip(lr_path_list, hr_path_list):
#         name = os.path.basename(lr_path) 
#         sr_save_path = os.path.join(save_dir, name)

#         lr_origin = cv2.imread(lr_path)
#         lr_RGB = cv2.cvtColor(lr_origin, cv2.COLOR_BGR2RGB)

#         hr_origin = cv2.imread(hr_path)
#         # hr_RGB = cv2.cvtColor(hr_origin, cv2.COLOR_BGR2RGB)

#         lr = _np2Tensor([lr_RGB])[0].unsqueeze(0).cuda()

#         weight = torch.ones(lr.size(0), LUT.LUT_tables_num, lr.size(2), lr.size(3)).cuda() / LUT.LUT_tables_num

#         sr = LUT(weight, lr, 4)
#         # print(sr.shape)
#         sr_img = sr.squeeze(0).detach().cpu().numpy().transpose((1,2,0))[:,:,(2,1,0)]
#         cv2.imwrite(sr_save_path, sr_img)

#         psnr, ssim = measure_single_path_wo_lpips(hr_path, sr_save_path, use_gpu=True)
#         print('file {}, psnr: {}, ssim: {}'.format(sr_save_path, psnr, ssim))
#         psnr_list.append(psnr)
#         ssim_list.append(ssim)
        
#     print('ave psnr: {}, ave ssim: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))

#inference time
if __name__ == '__main__':
    import time
    import tqdm
    LUT_file = '/home/v-gyin/github/VSR_LUT_hidden/LUT_tables/'
    tri_index_file = '/home/v-gyin/github/VSR_LUT_hidden/triangular_index.npy'
    LUT = Weighted4DLUTInter(LUT_file=LUT_file, tri_index_file=tri_index_file).cuda()

    LUT.eval()
    timeall = 0
    lr = torch.rand((1,3, 480, 320)).cuda()
    weight = torch.ones(lr.size(0), LUT.LUT_tables_num, lr.size(2), lr.size(3)).cuda() / LUT.LUT_tables_num
    repetitions = 240
    torch.cuda.synchronize()
    time0 = time.time()
    with torch.no_grad():
        for i in range(repetitions):
        
            sr = LUT(weight, lr, 4)
    torch.cuda.synchronize()
    time1 = time.time()
    timed = time1 - time0
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    print("warm up avg time: ", timed*1000/repetitions, " ms")

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    repetitions = 10000
    timings = np.zeros((repetitions, 1))

    torch.cuda.synchronize() # 等待GPU任务完成
    print('testing ...\n')
    # lr = torch.rand((1,3, 320, 180)).cuda()
    starter.record()
    with torch.no_grad():
        for rep in range(repetitions):
            #model running
            sr = LUT(weight, lr, 4)
            
    torch.cuda.synchronize() # 等待GPU任务完成
    ender.record()
    curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒

    avg = curr_time/repetitions
    print('\navg={} ms\n'.format(avg))
    #16ms

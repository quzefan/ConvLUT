import math
import cv2
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import quadrilinear4d
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

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

        # for p in self.parameters():
        #     p.requires_grad = False

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
        
        # weight_grad = torch.zeros([weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]], dtype=weight.dtype).to(weight.device)
        # print(weight_grad)

        assert 1 == quadrilinear4d.backward(lut,
                                      tri_index,
                                      weight,
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
        print(weight_grad)
        return None, None, weight_grad/16/3, None, None

class QuadrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(QuadrilinearInterpolation, self).__init__()

    def forward(self, lut, tri_index, weight, x, scale_factor):
        return QuadrilinearInterpolationFunction.apply(lut, tri_index, weight, x, scale_factor)

class PixelWeightPredictor(nn.Module):
    def __init__(self, LUT_num=2):
        super(PixelWeightPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=5//2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=5//2)
        self.conv4 = nn.Conv2d(64, LUT_num, 5, padding=5//2)
        self.relu = nn.LeakyReLU(0.2)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.in3 = nn.InstanceNorm2d(64, affine=True)
        self.sigmoid = nn.Sigmoid()

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('in') != -1 or classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        x = self.in1(self.relu(self.conv1(x_in)))
        x = self.in2(self.relu(self.conv2(x)))
        x = self.in3(self.relu(self.conv3(x)))
        x = self.conv4(x)
        x = self.sigmoid(x)

        return x

    def _load_pretrain_net(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        
class BiasPredictor(nn.Module):
    def __init__(self, in_channel=6):
        super(BiasPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(64, 3, 3, stride=1, padding=1, dilation=1)
        
        self.relu = nn.LeakyReLU(0.2)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('in') != -1 or classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x):
        # x = torch.cat((x_in, x_ref), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        y = self.conv4(x)

        return y

    def _load_pretrain_net(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class BiasPredictor_small(nn.Module):
    def __init__(self, num_channels=9, scale_factor=4, d=56, s=12, m=2):
        super(BiasPredictor_small, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, 3, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def _load_pretrain_net(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

if __name__ == '__main__':
    import time
    import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LUT_file = './LUT_tables/'
    tri_index_file = './triangular_index.npy'
    model_w = PixelWeightPredictor().to(device)
    model_b = BiasPredictor().to(device)
    LUT_interpolation = Weighted4DLUTInter(LUT_file=LUT_file, tri_index_file=tri_index_file).to(device)

    x = torch.rand((1, 3, 320, 180)).to(device)
    x_up = torch.rand((32, 3, 1280, 720)).to(device)

    model_w.eval()
    model_b.eval()
    LUT_interpolation.eval()

    timeall = 0
    repetitions = 240
    with torch.no_grad():
        for i in range(repetitions):
        
            time0 = time.time()
            weight = model_w(x)
            bias = model_b(torch.cat((x_up, x_up, x_up), dim=1))
            sr = LUT_interpolation(weight, x, 4) + bias
            time1 = time.time()
            timed = time1 - time0
            timeall = timeall + timed
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()
    print("warm up avg time: ", timeall*1000/repetitions, " ms")

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    repetitions = 100000
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    # lr = torch.rand((1,3, 320, 180)).cuda()
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for rep in range(repetitions):
            #model running
            weight = model_w(x)
            bias = model_b(torch.cat((x_up, x_up, x_up), dim=1))
            sr = LUT_interpolation(weight, x, 4) + bias
        torch.cuda.synchronize()
        end = time.time()
    print('Runtime: {} ms'.format((end-start)*1000/repetitions))

import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import numpy as np
import os

from SRResNet import SRResNet



if __name__ == '__main__':
    SCALE = 4
    SAMPLING_INTERVAL = 4        # N bit uniform sampling

    sr_name = 'SRResNet'
    qp=50
    model = SRResNet()
    # model._load_pretrain_net(torch.load('./SR_NTIRE_pretrain/SRResNet_saved_model.pth'))
    model._load_pretrain_net(torch.load('/home/v-gyin/github/SISR_LUT_weight_bias_sing/SRResNet_SR_models/SRResNet_qp{}_best_saved_model.pth'.format(qp)))

    model = model.cuda()

    ### Extract input-output pairs
    with torch.no_grad():
        model.eval()
        
        base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
        base[-1] -= 1
        L = base.size(0)
        
        # 2D input
        first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
        second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
        onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

        # 3D input
        third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
        onebytwo = onebytwo.repeat(L, 1)
        onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]

        # 4D input
        fourth = base.cuda().unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
        onebythree = onebythree.repeat(L, 1)
        onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]

        # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
        # input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float() / 255.0
        input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float()
        input_tensor = input_tensor.repeat(1, 3, 1, 1)
        print("Input size: ", input_tensor.size())

        # Split input to not over GPU memory
        B = input_tensor.size(0) // 100
        outputs = []

        for b in range(100):
            if b == 99:
                batch_output = model(input_tensor[b*B:])
            else:
                batch_output = model(input_tensor[b*B:(b+1)*B])
            batch_output = batch_output[:, 0, :, :].unsqueeze(1)

            # results = torch.round(torch.clamp(batch_output, -1, 1)*127).cpu().data.numpy().astype(np.int8)
            # results = torch.round(torch.clamp(batch_output*255, 0, 255)).cpu().data.numpy().astype(np.uint8)
            results = torch.round(torch.clamp(batch_output, 0, 255)).cpu().data.numpy().astype(np.uint8)
            outputs += [ results ]

        results = np.concatenate(outputs, 0)
        print("Resulting size of full_patch LUT: ", results.shape)
        # np.save("./LUT_tables/{}_x{}_4D_8x8_{}bit_uint8".format(sr_name, SCALE, SAMPLING_INTERVAL), results)
        results_LU = results[:,:,:SCALE,:SCALE]
        # results_RU = results[:,:,:SCALE,SCALE:]
        # results_LD = results[:,:,SCALE:,:SCALE]
        # results_RD = results[:,:,SCALE:,SCALE:]
        print("Resulting size of LU LUT: ", results_LU.shape)
        # print("Resulting size of RU LUT: ", results_RU.shape)
        # print("Resulting size of LD LUT: ", results_LD.shape)
        # print("Resulting size of RD LUT: ", results_RD.shape)

        np.save("/home/v-gyin/github/SISR_LUT_weight_bias_sing/SRResNet_LUT_tables/qp{}_{}_x{}_4D_LU_{}bit_uint8".format(qp, sr_name, SCALE, SAMPLING_INTERVAL), results_LU)
        # np.save("./{}_x{}_4D_RU_{}bit_uint8".format(sr_name, SCALE, SAMPLING_INTERVAL), results_RU)
        # np.save("./{}_x{}_4D_LD_{}bit_uint8".format(sr_name, SCALE, SAMPLING_INTERVAL), results_LD)
        # np.save("./{}_x{}_4D_RD_{}bit_uint8".format(sr_name, SCALE, SAMPLING_INTERVAL), results_RD)

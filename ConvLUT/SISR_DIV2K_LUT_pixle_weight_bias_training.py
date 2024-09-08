import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd

from option_SISR_DIV2K import set_args, check_args
from dataloader_SISR import LUTPixelDataSet, combine, crop_cpu
from model import PixelWeightPredictor, Weighted4DLUTInter, BiasPredictor, CharbonnierLoss, QuadrilinearInterpolation
from fusion_LUT_GPU_SR_parallel import SR_4DLUT_patch
from measure import measure_single_path_wo_lpips

import glob
import os
from copy import deepcopy
import random
import cv2
import numpy as np
from copy import deepcopy

def main():
    args = set_args()
    args = check_args(args)

    #to device finishing during model initial
    device = torch.device('cuda' if args.gpu_ids is not None else 'cpu')

    #dataloader
    train_dataset = LUTPixelDataSet(args, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True)
    test_dataset = LUTPixelDataSet(args, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    LUT_list = glob.glob(os.path.join(args.lut_tables_path, '*.npy'))
    LUT_tables = []
    for LUT_path in LUT_list:
        LUT_table = np.load(LUT_path).astype(np.float32).reshape(-1, args.scale_factor*args.scale_factor)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
        LUT_tables.append(LUT_table)
    LUT_tables_num = len(LUT_tables)
    if LUT_tables_num == 0:
        raise RuntimeError('LUT table load error!! Empty')
    LUT_tensor = torch.from_numpy(np.array(LUT_tables)).to(device)

    dim, scale_factor = 17, 4
    if args.lut_tables_path == None:
        LUTs = torch.ones(1, dim,dim,dim,dim,scale_factor,scale_factor, dtype=torch.float).to(device)
    else:
        #load all npy from LUT_file
        LUT_list = glob.glob(os.path.join(args.lut_tables_path, '*.npy'))
        LUT_tables = []
        for LUT_path in LUT_list:
            LUT_table = np.load(LUT_path).astype(np.float32).reshape(dim,dim,dim,dim,scale_factor,scale_factor)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
            LUT_tables.append(LUT_table)
        LUT_tables_num = len(LUT_tables)
        if LUT_tables_num == 0:
            raise RuntimeError('LUT table load error!! Empty')
        LUTs = torch.from_numpy(np.array(LUT_tables)).to(device)
    # lut, tri_index, weight, x, scale_factor
    if args.triangular_index_path == None:
        raise ValueError("invalid tri_index_file path!!")
    tri_index = np.load(args.triangular_index_path).astype(np.float32)
    tri_index = torch.from_numpy(np.array(tri_index)).to(device)

    # LUT_interpolater = Weighted4DLUTInter(LUT_file=args.lut_tables_path, tri_index_file=args.triangular_index_path).to(device)
    LUT_interpolater = QuadrilinearInterpolation().to(device)
    model_w = PixelWeightPredictor(LUT_num=LUT_tables_num).to(device)
    model_b = BiasPredictor(3).to(device)
    # model_w._load_pretrain_net(torch.load('/home/v-gyin/github/VSR_LUT_main/checkpoint_SISR_DIV2K/models/Weight_LUT_predictor_best_saved_model.pth'))
    # model_b._load_pretrain_net(torch.load('/home/v-gyin/github/VSR_LUT_main/checkpoint_SISR_DIV2K/models/Bias_LUT_predictor_best_saved_model.pth'))
    #loss
    train_pixel_loss = torch.nn.L1Loss()
    # train_pixel_loss = CharbonnierLoss()
    #optimizer
    # params_W = list(filter(lambda p: p.requires_grad, model_w.parameters()))
    # params_B = list(filter(lambda p: p.requires_grad, model_b.parameters()))
    # train_optimizer = torch.optim.Adam([{'params':params_W},{'params':params_LUT},{'params':params_B}], lr = args.lr)

    train_w_optimizer = torch.optim.Adam(model_w.parameters(), lr = args.lr*0.1)
    train_b_optimizer = torch.optim.Adam(model_b.parameters(), lr = args.lr)

    best_psnr, best_ssim = 0, 0

    #run/time
    with open(args.log_file, 'w') as f:
        print('The batch nums of training support loader {:,d}'.format(len(train_dataset)), file=f)
        print('The batch nums of training support loader {:,d}'.format(len(train_dataset)))
    for epoch in range(args.epochs):
        current_step = 0
        with open(args.log_file, 'a+') as f:
            print('Epoch : {}'.format(epoch), file = f)
            print('Epoch : {}'.format(epoch))
        model_w.train()
        # model_b.train()
        for lr_patches, lr_up_patches, hr_patches, lr_filenames, hr_filenames in train_loader:
            train_w_optimizer.zero_grad()
            train_b_optimizer.zero_grad()
            # train_optimizer.zero_grad()
            
            # for weight prediction
            # [image_batch, 3, 32, 32],  [image_batch, 3, 128, 128]
            # for SR with LUT
            # weight prediction for each pixel
            # lr_patches, lr_up_patches, hr_patches = lr_patches.to(device), lr_up_patches.to(device), hr_patches.to(device)
            # weights = model_w(lr_patches) #[image_batch, num_LUT, 32, 32]
            # # biases = model_b(lr_up_patches) #[image_batch, 3, 128, 128]
            # _, sr_weighted_patches = LUT_interpolater(LUTs, tri_index, weights, lr_patches, 4)
            # sr_patches = sr_weighted_patches
            # # sr_patches = sr_weighted_patches + biases
            # sr_loss1 = train_pixel_loss(sr_patches, hr_patches)
            # sr_loss1.backward()
            # train_w_optimizer.step()

            weights = model_w(lr_patches) #[image_batch, num_LUT, 32, 32]
            # _, sr_weighted_patches = LUT_interpolater(LUTs, tri_index, weights, lr_patches, 4)
            sr_weighted_patches = SR_4DLUT_patch(lr_patches, LUT_tensor, weights1, args.scale_factor, args.sampling_interval)
            # sr_patches = sr_weighted_patches
            sr_patches = sr_weighted_patches + biases
            sr_loss = train_pixel_loss(sr_patches, hr_patches)
            # weight_grad = autograd.grad(sr_loss, weights1, retain_graph=True)[0]
            # print(weight_grad)
            sr_loss.backward()

            train_w_optimizer.step()
            train_b_optimizer.step()
            if current_step % args.print_train_loss_every == 0:
                with open(args.log_file, 'a+') as f:
                    print('Epoch:{} Step:{} joint SR_LUT loss:{} '.format(epoch, current_step, sr_loss.cpu().detach().item()))
                    print('Epoch:{} Step:{} joint SR_LUT loss:{} '.format(epoch, current_step, sr_loss.cpu().detach().item()), file = f)
                    weight_loss = train_pixel_loss(sr_weighted_patches.cpu().detach(), hr_patches.cpu().detach())
                    print('Epoch:{} Step:{} weight SR_LUT loss:{} '.format(epoch, current_step, weight_loss.cpu().detach().item()))
                    print('Epoch:{} Step:{} weight SR_LUT loss:{} '.format(epoch, current_step, weight_loss.cpu().detach().item()), file = f)
                    bias_loss = train_pixel_loss(biases.cpu().detach(), hr_patches.cpu().detach())
                    print('Epoch:{} Step:{} bias SR_LUT loss:{} '.format(epoch, current_step, bias_loss.cpu().detach().item()))
                    print('Epoch:{} Step:{} bias SR_LUT loss:{} '.format(epoch, current_step, bias_loss.cpu().detach().item()), file = f)
            current_step += 1
            
        #Test
        if epoch > 50 and epoch % args.run_test_every == 0:
            psnr_list, ssim_list = [], []
            result_checkpoint_dir = os.path.join(args.result_checkpoint_dir, 'Epoch_{}'.format(epoch))
            if not os.path.exists(result_checkpoint_dir):
                os.mkdir(result_checkpoint_dir)

            if args.self_ensemble_output:
                for lr1, lr_up1, lr2, lr_up2, lr3, lr_up3, lr4, lr_up4, hr, lr_filename, hr_filename in test_loader: 
                    lr_list, lr_up_list, hr = [lr1.to(device), lr2.to(device), lr3.to(device), lr4.to(device)], [lr_up1.to(device), lr_up2.to(device), lr_up3.to(device), lr_up4.to(device)], hr.to(device)
                    test_sr_save_path = os.path.join(result_checkpoint_dir, lr_filename[0])
                    test_hr_save_path = os.path.join(args.test_hr_file_path, hr_filename[0])
                    
                    model_w.eval()
                    model_b.eval()
                    rot = 4
                    sr_self_ensemble_list = []
                    with torch.no_grad():
                        for lr, lr_up in zip(lr_list, lr_up_list):
                            weight = model_w(lr)
                            bias = model_b(lr_up)
                            sr = LUT_interpolater(LUTs, tri_index, weight, lr, 4)
                            sr = sr + bias
                            sr = sr.squeeze(0).detach().cpu().numpy()
                            sr = np.rot90(sr, rot, [1,2]).transpose((1,2,0))[:,:,(2,1,0)]
                            rot -= 1
                            sr_self_ensemble_list.append(sr)
                    sr = (sr_self_ensemble_list[0] + sr_self_ensemble_list[1] + sr_self_ensemble_list[2] + sr_self_ensemble_list[3]) / 4.0
                    
                    cv2.imwrite(test_sr_save_path, sr)
                    psnr, ssim = measure_single_path_wo_lpips(test_hr_save_path, test_sr_save_path, use_gpu=True)
                    with open(args.log_file, 'a+') as f:
                        print('Img {}, psnr: {}, ssim: {}'.format(test_sr_save_path, psnr, ssim))
                        print('Img {}, psnr: {}, ssim: {}'.format(test_sr_save_path, psnr, ssim), file = f)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    
            else:
                for lr, lr_up, hr, lr_filename, hr_filename in test_loader:
                    lr, lr_up, hr = lr.to(device), lr_up.to(device), hr.to(device)
                    test_sr_save_path = os.path.join(result_checkpoint_dir, lr_filename[0])
                    test_hr_save_path = os.path.join(args.test_hr_file_path, hr_filename[0])
                    
                    model_w.eval()
                    model_b.eval()
                    with torch.no_grad():
                        weight = model_w(lr)
                        bias = model_b(lr_up)
                        sr = LUT_interpolater(LUTs, tri_index, weight, lr, 4)
                        sr = sr + bias
                    
                        sr = sr.squeeze(0).detach().cpu().numpy().transpose((1,2,0))[:,:,(2,1,0)]
                
                        cv2.imwrite(test_sr_save_path, sr)

                        psnr, ssim = measure_single_path_wo_lpips(test_hr_save_path, test_sr_save_path, use_gpu=False)
                        with open(args.log_file, 'a+') as f:
                            print('Img {}, psnr: {}, ssim: {}'.format(test_sr_save_path, psnr, ssim))
                            print('Img {}, psnr: {}, ssim: {}'.format(test_sr_save_path, psnr, ssim), file = f)
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
            ave_psnr, ave_ssim = np.mean(psnr_list), np.mean(ssim_list)
            if ave_psnr > best_psnr or ave_ssim > best_ssim:
                best_psnr, best_ssim = ave_psnr, ave_ssim
                torch.save(model_w.state_dict(), os.path.join(args.model_checkpoint_dir, 'Weight_LUT_predictor_best_saved_model.pth'))
                torch.save(model_b.state_dict(), os.path.join(args.model_checkpoint_dir, 'Bias_LUT_predictor_best_saved_model.pth'))
                with open(args.log_file, 'a+') as f:
                    print('New best in Epoch {} ave psnr:{}, ave ssim:{}'.format(epoch, ave_psnr, ave_ssim))
                    print('New best in Epoch {}'.format(epoch), file = f)
            with open(args.log_file, 'a+') as f: 
                print('Epoch {} ave psnr: {}, ave ssim: {}'.format(epoch, ave_psnr, ave_ssim))
                print('Epoch {} ave psnr: {}, ave ssim: {}'.format(epoch, ave_psnr, ave_ssim), file = f)

        if epoch % args.run_test_every == 0:
            torch.save(model_w.state_dict(), os.path.join(args.model_checkpoint_dir, 'Weight_LUT_predictor_'+str(epoch)+'_saved_model.pth'))
            torch.save(model_b.state_dict(), os.path.join(args.model_checkpoint_dir, 'Bias_LUT_predictor_'+str(epoch)+'_saved_model.pth'))


if __name__ == '__main__':
    main()

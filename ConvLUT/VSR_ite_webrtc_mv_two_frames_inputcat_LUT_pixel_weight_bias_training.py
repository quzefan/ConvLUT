import torch
from torch.utils.data import DataLoader

from option_2frames_webrtc import set_args, check_args
from dataloader_mv2frames import LUTVSRDataSet
from model import PixelWeightPredictor, Weighted4DLUTInter, BiasPredictor, CharbonnierLoss, BiasPredictor_small
from fusion_LUT_GPU_SR_parallel import SR_4DLUT_patch, crop_cpu, combine
from PSNR_SSIM_cal import cal_PSNR, cal_SSIM
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

    LUT_list = glob.glob(os.path.join(args.lut_tables_path, '*.npy'))
    LUT_tables = []
    for LUT_path in LUT_list:
        LUT_table = np.load(LUT_path).astype(np.float32).reshape(-1, args.scale_factor*args.scale_factor)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
        LUT_tables.append(LUT_table)
    LUT_tables_num = len(LUT_tables)
    if LUT_tables_num == 0:
        raise RuntimeError('LUT table load error!! Empty')
    LUT_tensor = torch.from_numpy(np.array(LUT_tables)).to(device)

    #dataloader
    train_dataset = LUTVSRDataSet(args, is_up=False, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    test_dataset = LUTVSRDataSet(args, is_up=False, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    LUT_interpolater = Weighted4DLUTInter(LUT_file=args.lut_tables_path, tri_index_file=args.triangular_index_path).to(device)
    model_w = PixelWeightPredictor(LUT_num=LUT_interpolater.LUT_tables_num).to(device)
    # model_b = BiasPredictor(9).to(device)
    model_b = BiasPredictor_small(num_channels=9).to(device)
    # model_w._load_pretrain_net(torch.load('./checkpoint_pixel_w_b_two_frames/models/Weight_LUT_predictor_best_saved_model.pth'))
    # model_b._load_pretrain_net(torch.load('./checkpoint_pixel_w_b_two_frames/models/Bias_LUT_predictor_best_saved_model.pth'))
    # model_w._load_pretrain_net(torch.load('/home/v-gyin/github/VSR_LUT_hidden/checkpoint_pixel_w_b_two_frames/models/Weight_LUT_predictor_best_saved_model.pth'))
    # model_b._load_pretrain_net(torch.load('/home/v-gyin/github/VSR_LUT_hidden/checkpoint_pixel_w_b_two_frames/models/Bias_LUT_predictor_best_saved_model.pth'))
    
    #loss
    train_pixel_loss = torch.nn.L1Loss()
    # train_pixel_loss = CharbonnierLoss()
    #optimizer
    params_W = list(filter(lambda p: p.requires_grad, model_w.parameters()))
    params_B = list(filter(lambda p: p.requires_grad, model_b.parameters()))
    # train_optimizer = torch.optim.Adam([{'params':params_W},{'params':params_B}], lr = args.lr)
    train_w_optimizer = torch.optim.Adam(model_w.parameters(), lr = args.lr)
    train_b_optimizer = torch.optim.Adam(model_b.parameters(), lr = args.lr)

    best_psnr, best_ssim = 0, 0
    joint_train = False

    #run/time
    with open(args.log_file, 'w') as f:
        print('The batch nums of training support loader {:,d}'.format(len(train_dataset)), file=f)
        print('The batch nums of training support loader {:,d}'.format(len(train_dataset)))
    current_step = 0
    for epoch in range(args.epochs):
        with open(args.log_file, 'a+') as f:
            print('Epoch : {}'.format(epoch), file = f)
            print('Epoch : {}'.format(epoch))
        for lr_patches, lr_up_patches, lr_ref_up_patches, lr_mv_up_patches, hr_patches, lr_pathes, hr_pathes in train_loader:
            #[b,1,3,h,w], [b,2,3,h,w], [b,1,3,h,w]
            b,c,h,w = hr_patches.shape
            model_w.train()
            model_b.train()
            hr_patches = hr_patches.to(device)
            train_w_optimizer.zero_grad()
            train_b_optimizer.zero_grad()
            
            lr_patches, lr_up_patches, lr_ref_up_patches, lr_mv_up_patches, hr_patches = lr_patches.to(device), lr_up_patches.to(device), lr_ref_up_patches.to(device), lr_mv_up_patches.to(device), hr_patches.to(device)
            weights = model_w(lr_patches) #[image_batch, num_LUT, h, w]
            biases_patches = model_b(torch.cat((lr_up_patches, lr_ref_up_patches, lr_mv_up_patches), dim=1)) #[image_batch, 3, 128, 128]
            sr_LUT_patches = SR_4DLUT_patch(lr_patches, LUT_tensor, weights, args.scale_factor, args.sampling_interval)
            # sr_LUT_patches = LUT_interpolater(weights, lr_patches, 4)

            sr_patches = sr_LUT_patches + biases_patches
                
            weight_loss = train_pixel_loss(sr_LUT_patches.detach().cpu(), hr_patches.detach().cpu())
            bias_loss = train_pixel_loss(biases_patches.detach().cpu(), hr_patches.detach().cpu() - sr_LUT_patches.detach().cpu())
            sr_loss = train_pixel_loss(sr_patches, hr_patches)
            sr_loss.backward()
            train_w_optimizer.step()
            train_b_optimizer.step()
            
            if current_step % args.print_train_loss_every == 0:
                with open(args.log_file, 'a+') as f:
                    print('Epoch:{} Step:{} joint SR_LUT loss:{} '.format(epoch, current_step, sr_loss.cpu().detach().item()))
                    print('Epoch:{} Step:{} joint SR_LUT loss:{} '.format(epoch, current_step, sr_loss.cpu().detach().item()), file = f)
                    print('Epoch:{} Step:{} weight SR_LUT loss:{} '.format(epoch, current_step, weight_loss.cpu().detach().item()))
                    print('Epoch:{} Step:{} weight SR_LUT loss:{} '.format(epoch, current_step, weight_loss.cpu().detach().item()), file = f)
                    print('Epoch:{} Step:{} bias SR_LUT loss:{} '.format(epoch, current_step, bias_loss.cpu().detach().item()))
                    print('Epoch:{} Step:{} bias SR_LUT loss:{} '.format(epoch, current_step, bias_loss.cpu().detach().item()), file = f)
            torch.cuda.empty_cache()
            current_step += 1
        
            #Test
            # if epoch > 5 and epoch % args.run_test_every == 0:
            if (current_step-1) % args.run_test_every_itera == 0:
            # if current_step % args.run_test_every_itera == 0:
                psnr_list, ssim_list = [], []
                result_checkpoint_dir = os.path.join(args.result_checkpoint_dir, 'Iter_{}'.format(current_step))
                if not os.path.exists(result_checkpoint_dir):
                    os.mkdir(result_checkpoint_dir)

                for lr, lr_up, lr_ref_up, lr_mv_up, hr, lr_pathes, hr_pathes in test_loader:
                    b,c,h,w = hr.shape
                    lr, lr_up, lr_ref_up, lr_mv_up, hr = lr.to(device), lr_up.to(device), lr_ref_up.to(device), lr_mv_up.to(device), hr.to(device)
                    lr_filename = os.path.basename(lr_pathes[0])
                    test_sr_save_path = os.path.join(result_checkpoint_dir, lr_filename)
                    test_hr_save_path = hr_pathes[0]
                        
                    model_w.eval()
                    model_b.eval()
                    with torch.no_grad():
                        weight = model_w(lr)
                        bias = model_b(torch.cat((lr_up, lr_ref_up, lr_mv_up), dim=1))
                        sr = LUT_interpolater(weight, lr, 4)
                        sr = sr + bias
                        sr_pre = sr
                        sr_img = sr.squeeze(0).detach().cpu().numpy().transpose((1,2,0))[:,:,(2,1,0)]
                        hr_img = hr.squeeze(0).detach().cpu().numpy().transpose((1,2,0))[:,:,(2,1,0)]
                
                        # cv2.imwrite(test_sr_save_path, sr_img)
                        psnr, ssim = cal_PSNR(sr_img, hr_img), cal_SSIM(sr_img, hr_img, False)
                        # psnr, ssim = measure_single_path_wo_lpips(test_hr_save_path, test_sr_save_path, use_gpu=False)
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

            if current_step % args.run_test_every_itera == 0:
                torch.save(model_w.state_dict(), os.path.join(args.model_checkpoint_dir, 'Weight_LUT_predictor_'+str(current_step)+'_saved_model.pth'))
                torch.save(model_b.state_dict(), os.path.join(args.model_checkpoint_dir, 'Bias_LUT_predictor_'+str(current_step)+'_saved_model.pth'))


if __name__ == '__main__':
    main()
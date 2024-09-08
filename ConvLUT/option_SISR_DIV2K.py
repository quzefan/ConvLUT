import argparse
import os
import glob

"""
Configuration file
"""
def check_args(args, rank=0):
    if rank == 0:
        with open(args.setting_file, 'w') as opt_file:
            opt_file.write('------------ Options -------------/n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')

    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--is_train', type=str2bool, default=True)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--batch_size_inner_patch', type=int, default=8, help='patch size for LUT ')
    parser.add_argument('--epochs', type=int, default=10000, help='Epochs for training')
    #define for data preprocessing
    parser.add_argument('--is_Ycbcr', type=str2bool, default= False)
    parser.add_argument('--random_crop', type=str2bool, default=True)
    parser.add_argument('--rgb_range', type=int, default=255, help='RGB range for input image')
    
    #sr_net
    parser.add_argument('--input_channels', type=int, default=3, help='the number of input channels for sr net')
    parser.add_argument('--channels', type=int, default=64, help='the number of hidden channels for sr net')
    parser.add_argument('--residual_lr', type=float, default=1.0, help='the lr coefficient of residual connection')
    parser.add_argument('--kernel_size', type=int, default=3, help='the kernel_size of conv')
    parser.add_argument('--n_block', type=int, default=10, help='the number of res-block')

    #lut
    parser.add_argument('--patch_pixel_size', type=int, default=48, help='Crop size for pixel weight and bias prediction')
    parser.add_argument('--sampling_interval', type=int, default=4)

    #define for pre-training
    parser.add_argument('--load_pretrained_model', type=str2bool, default=False)
    parser.add_argument('--load_pretrained_model_path', type=str, default='')

    parser.add_argument('--net_name', type=str, default='SR_LUT_weight')
    parser.add_argument('--save_metanet_every', type=int, default=5000)

    # define for validation
    parser.add_argument('--print_train_loss_every', type=int, default=10)
    parser.add_argument('--save_train_loss_every', type=int, default=1)
    parser.add_argument('--run_test_every', type=int, default=10)
    parser.add_argument('--start_save', type=int, default=100000)
    parser.add_argument('--save_test_psnr_ssim_every', type=int, default=1)
    parser.add_argument('--self_ensemble_output', type=str2bool, default=False, help='geometric self-ensemble')
    parser.add_argument('--flip_output', type=str2bool, default=False, help='conduct flip for geometric self-ensemble')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--step_overlap', type=int, default=6)
    
    #checkpoint
    parser.add_argument('--sing', type=str2bool,
                        default=False)
    parser.add_argument('--lut_tables_path', type=str,
                        default='/LUT_tables_BI/')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/checkpoint_SISR_DIV2K')
    parser.add_argument('--triangular_index_path', type=str,
                        default='/triangular_index.npy')
    # parser.add_argument('--checkpoint_dir', type=str,
                        # default='/home/v-gyin/github/RCAN_LUT_weighted/checkpoint_pixel_w_b')
    parser.add_argument('--train_hr_file_path', type=str,
                            default='/dataset/DIV2k/DIV2K_train_HR')
    parser.add_argument('--train_lr_file_path', type=str,
                            default='/dataset/DIV2k/DIV2K_train_LR_bicubic/X4/')
    parser.add_argument('--test_hr_file_path', type=str,
                            default='/dataset/Set5/Set5_HR/')
    parser.add_argument('--test_lr_file_path', type=str,
                            default='/dataset/Set5/Set5_BIx4/')
    parser.add_argument('--checkpoint_sub_dir', type=str, default='Train')

    args = parser.parse_args()
    if args.sing:
        data_root_path = '.'
        args.lut_tables_path = data_root_path + args.lut_tables_path
        args.triangular_index_path = data_root_path + args.triangular_index_path
        args.checkpoint_dir = data_root_path + args.checkpoint_dir
    else:
        data_root_path = '/home/v-gyin'
        args.lut_tables_path = data_root_path + '/github/VSR_LUT_main' + args.lut_tables_path
        args.triangular_index_path = data_root_path + '/github/VSR_LUT_main' + args.triangular_index_path
        args.checkpoint_dir =data_root_path + '/github/VSR_LUT_main' + args.checkpoint_dir
    
    args.train_hr_file_path = data_root_path + args.train_hr_file_path
    args.train_lr_file_path = data_root_path + args.train_lr_file_path
    args.test_hr_file_path = data_root_path + args.test_hr_file_path
    args.test_lr_file_path = data_root_path + args.test_lr_file_path

    #set save losses, psnr, ssim plt
    args.save_train_loss_every = args.save_train_loss_every * args.print_train_loss_every
    args.save_test_psnr_ssim_every = args.save_test_psnr_ssim_every * args.run_test_every
    
    if not os.path.exists(args.train_hr_file_path):
        raise ValueError("Misssing %s"%args.train_hr_file_path)
    else:
        print('%s exists!'%args.train_hr_file_path)
    if not os.path.exists(args.train_lr_file_path):
      raise ValueError("Misssing %s"%args.train_lr_file_path)
    else:
        print('%s exists!'%args.train_lr_file_path)
    if not os.path.exists(args.test_hr_file_path):
        raise ValueError("Misssing %s"%args.test_hr_file_path)
    else:
        print('%s exists!'%args.test_hr_file_path)
    if not os.path.exists(args.test_hr_file_path):
        raise ValueError("Misssing %s"%args.test_hr_file_path)
    else:
        print('%s exists!'%args.test_hr_file_path)
    
    # args.checkpoint_dir += args.checkpoint_sub_dir
    args.model_checkpoint_dir = args.checkpoint_dir + '/models/'
    args.result_checkpoint_dir = args.checkpoint_dir + '/results/'
    args.pretrained_model_checkpoint_dir = args.checkpoint_dir + '/pretrained_models/'
    args.log_checkpoint_dir = args.checkpoint_dir + '/log/'
    args.setting_file = args.checkpoint_dir + '/' + args.setting_file
    args.log_file = args.checkpoint_dir + '/' + 'log.txt'
    #create checkpoint dirs
    if os.path.exists(args.checkpoint_dir) and os.path.isfile(args.checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint saving, got a file'.format(
          args.checkpoint_dir))
    elif not os.path.exists(args.checkpoint_dir):
      os.makedirs(args.checkpoint_dir)
      print('%s created successfully!'%args.checkpoint_dir)

    if os.path.exists(args.pretrained_model_checkpoint_dir) and os.path.isfile(args.pretrained_model_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint pretrained model saving, got a file'.format(
          args.pretrained_model_checkpoint_dir))
    elif not os.path.exists(args.pretrained_model_checkpoint_dir):
      os.makedirs(args.pretrained_model_checkpoint_dir)
      print('%s created successfully!'%args.pretrained_model_checkpoint_dir)
    
    if os.path.exists(args.result_checkpoint_dir) and os.path.isfile(args.result_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint results saving, got a file'.format(
          args.result_checkpoint_dir))
    elif not os.path.exists(args.result_checkpoint_dir):
      os.makedirs(args.result_checkpoint_dir)
      print('%s created successfully!'%args.result_checkpoint_dir)
    
    if os.path.exists(args.model_checkpoint_dir) and os.path.isfile(args.model_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint model saving, got a file'.format(
          args.model_checkpoint_dir))
    elif not os.path.exists(args.model_checkpoint_dir):
      os.makedirs(args.model_checkpoint_dir)
      print('%s created successfully!'%args.model_checkpoint_dir)

    if os.path.exists(args.model_checkpoint_dir) and os.path.isfile(args.log_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint log saving, got a file'.format(
          args.log_checkpoint_dir))
    elif not os.path.exists(args.log_checkpoint_dir):
      os.makedirs(args.log_checkpoint_dir)
      print('%s created successfully!'%args.log_checkpoint_dir)
    
    if args.load_pretrained_model:
        args.load_pretrained_model_path = os.path.join(args.pretrained_model_checkpoint_dir, load_pretrained_model_path)
    return args

if __name__ == "__main__":
    args = set_args()   

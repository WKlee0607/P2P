# -*- coding: utf-8 -*-
"""
Options for the pix2pixCC model

"""

#==============================================================================

import os
import argparse


#==============================================================================

class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--gpu_ids', type=str, default="3", help='gpu number. If -1, use cpu')

        #----------------------------------------------------------------------
        # data setting
        
        self.parser.add_argument('--dataset_name', type=str, default='SAT2Rain', help='dataset directory name')
        self.parser.add_argument('--data_format_input', type=str, default='npy', help="Input data extension. [tif, tiff, npy, fits, fts, fit]")
        self.parser.add_argument('--data_format_target', type=str, default='npy', help="Target data extension. [tif, tiff, npy, fits, fts, fit]")

        self.parser.add_argument('--input_ch', type=int, default=3, help="The number of channels of input data") # 0:vi, 1:ir, 2: wv
        self.parser.add_argument('--target_ch', type=int, default=1, help="The number of channels of target data")

        self.parser.add_argument('--data_size', type=int, default=256, help='image size of the input and target data') # 

        self.parser.add_argument('--logscale_input', type=bool, default=False, help='use logarithmic scales to the input data sets')
        self.parser.add_argument('--logscale_target', type=bool, default=False, help="use logarithmic scales to the target data sets")
        self.parser.add_argument('--saturation_lower_limit_input', type=float, default=-1, help="Saturation value (lower limit) of input")
        self.parser.add_argument('--saturation_upper_limit_input', type=float, default=1, help="Saturation value (upper limit) of input")
        self.parser.add_argument('--saturation_lower_limit_target', type=float, default=-1, help="Saturation value (lower limit) of target")
        self.parser.add_argument('--saturation_upper_limit_target', type=float, default=1, help="Saturation value (upper limit) of target")
        self.parser.add_argument('--saturation_clip_input', type=bool, default=False, help="Saturation clip for input data")
        self.parser.add_argument('--saturation_clip_target', type=bool, default=False, help="Saturation clip for target data")


        #----------------------------------------------------------------------
        # network setting
        # 모델 파라미터 변경시 수정 
        self.parser.add_argument('--batch_size', type=int, default=64, help='the number of batch_size') # 64

        self.parser.add_argument('--n_downsample', type=int, default=4, help='how many times you want to downsample input data in Generator')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in Generator')
        self.parser.add_argument('--trans_conv', type=bool, default=True, help='using transposed convolutions in Generator')

        self.parser.add_argument('--n_D', type=int, default=3, help='how many Discriminators in differet scales you want to use') # 1
        self.parser.add_argument('--n_CC', type=int, default=4, help='how many downsample output data to compute CC values')

        self.parser.add_argument('--n_gf', type=int, default=64, help='The number of channels in the first convolutional layer of the Generator')
        self.parser.add_argument('--n_df', type=int, default=64, help='The number of channels in the first convolutional layer of the Discriminator')

        self.parser.add_argument('--data_type', type=int, default=32, help='float dtype [16, 32]')
        self.parser.add_argument('--n_workers', type=int, default=10, help='how many threads you want to use')

        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d', help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--padding_type', type=str, default='reflection',  help='[reflection, replication, zero]')

        #----------------------------------------------------------------------
        # data augmentation

        self.parser.add_argument('--padding_size', type=int, default=0, help='padding size for random crop')
        self.parser.add_argument('--max_rotation_angle', type=int, default=0, help='rotation angle in degrees')

        #----------------------------------------------------------------------
        # checking option
        # 훈련 시 체크 & 저장 옵션 수정 가능 
        self.parser.add_argument('--report_freq', type=int, default=10000) # 10000
        self.parser.add_argument('--save_freq', type=int, default=100000)
        self.parser.add_argument('--display_freq', type=int, default=10000) # 10000
        self.parser.add_argument('--save_scale', type=float, default=1)
        self.parser.add_argument('--display_scale', type=float, default=1)
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--val_during_train', action='store_true', default=True)

        #----------------------------------------------------------------------


    def parse(self):
        opt = self.parser.parse_args()
        opt.format = 'png'  # extension for checking image
        opt.flip = False

        #--------------------------------
        if opt.data_type == 16:
            opt.eps = 1e-4
        elif opt.data_type == 32:
            opt.eps = 1e-8

        #--------------------------------
        dataset_name = opt.dataset_name
        # 훈련 결과 저장 디렉토리 생성 
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Train'), exist_ok=True) #'Train'
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Test'), exist_ok=True) # 'Test'
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Model'), exist_ok=True)

        if opt.is_train:
            opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Train') # 'Image/Train'
        else:
            opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Test') # 'Image/Test'

        opt.model_dir = os.path.join('./checkpoints', dataset_name, 'Model')

        #--------------------------------
        return opt



#==============================================================================

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        #----------------------------------------------------------------------
        # directory path for trainin

        self.parser.add_argument('--input_dir_train', type=str, default='/local_datasets/SAT2Rain/daytime_model/train/input/', help='directory path of the input files for the model training')
        self.parser.add_argument('--target_dir_train', type=str, default='/local_datasets/SAT2Rain/daytime_model/train/target/', help='directory path of the target files for the model training')

        #----------------------------------------------------------------------
        # Train setting

        self.parser.add_argument('--is_train', type=bool, default=True, help='train flag')
        self.parser.add_argument('--n_epochs', type=int, default=120, help='how many epochs you want to train')
        self.parser.add_argument('--latest_iter', type=int, default=-1, help='Resume iteration')
        self.parser.add_argument('--no_shuffle', action='store_true', default=False, help='if you want to shuffle the order')

        #----------------------------------------------------------------------
        # hyperparameters (손실 함수 웨이트 조정  가능 )

        self.parser.add_argument('--lambda_LSGAN', type=float, default=2.0, help='weight for LSGAN loss')
        self.parser.add_argument('--lambda_FM', type=float, default=10.0, help='weight for Feature Matching loss')
        self.parser.add_argument('--lambda_CC', type=float, default=5.0, help='weight for CC loss')

        self.parser.add_argument('--ch_balance', type=float, default=1, help='Set channel balance of input and target data')
        self.parser.add_argument('--ccc', type=bool, default=True, help='using Concordance Correlation Coefficient values (False -> using Pearson CC values)')

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--epoch_decay', type=int, default=150, help='when to start decay the lr')
        self.parser.add_argument('--lr', type=float, default=0.0005) # 0.0002

        #----------------------------------------------------------------------



#==============================================================================

class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        #----------------------------------------------------------------------
        # directory path for test

        self.parser.add_argument('--input_dir_test', type=str, default='/local_datasets/SAT2Rain/daytime_model/test/input/', help='directory path of the input files for the model test')
        self.parser.add_argument('--target_dir_test', type=str, default='/local_datasets/SAT2Rain/daytime_model/test/target/', help='directory path of the target files for the model training')

        #----------------------------------------------------------------------
        # test setting

        self.parser.add_argument('--is_train', type=bool, default=False, help='test flag')
        self.parser.add_argument('--iteration', type=int, default=6500000, help='if you want to generate from input for the specific iteration')
        self.parser.add_argument('--no_shuffle', type=bool, default=True, help='if you want to shuffle the order')

        #----------------------------------------------------------------------



#==============================================================================

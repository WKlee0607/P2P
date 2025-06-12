"""
Pipeline of the pix2pixCC model
"""

#==============================================================================

import os
from os.path import split, splitext
import numpy as np
from glob import glob
from random import randint

import torchvision.transforms
from PIL import Image
from astropy.io import fits
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset

from pix2pixCC_Options import TrainOption
trainopt = TrainOption().parse()
#==============================================================================
# [1] Preparing the Input and Target data sets

class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        self.s = 290
        self.e = 200
                
        if opt.is_train:
            self.input_format = opt.data_format_input # format: npy
            self.target_format = opt.data_format_target # format: npy
            self.input_dir = opt.input_dir_train # pix2pixCC_Option.py - TrainOption class
            self.target_dir = opt.target_dir_train # pix2pixCC_Option.py - TrainOption class

            self.label_path_list = sorted(glob(os.path.join(self.input_dir, '*.' + self.input_format))) 
            self.target_path_list = sorted(glob(os.path.join(self.target_dir, '*.' + self.target_format)))
            print(len(self.label_path_list), len(self.target_path_list))

        else: # test mode
            self.input_format = opt.data_format_input # format: npy
            self.target_format = opt.data_format_target # format: npy
            self.input_dir = opt.input_dir_test # pix2pixCC_Option.py - TestOption class
            self.target_dir = opt.target_dir_test

            self.label_path_list = sorted(glob(os.path.join(self.input_dir, '*.' + self.input_format)))
            self.target_path_list = sorted(glob(os.path.join(self.target_dir, '*.' + self.target_format)))
            print(len(self.label_path_list), len(self.target_path_list))
            

    def __getitem__(self, index):
        list_transforms = []
        list_transforms += []



        # [ Train data ] ==============================================================
        if self.opt.is_train: #data argumentation
            self.angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle) #defalut:0 --> 회전변환x

            self.offset_x = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            self.offset_y = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            
# [ Train Input ] =============================================================
            if self.input_format in ["tif", "tiff"]:
                IMG_A0 = np.array(Image.open(self.label_path_list[index]))
            elif self.input_format in ["npy"]: # 데이터 포맷:넘파이 
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            elif self.input_format in ["fits", "fts", "fit"]:
                IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data)
            else:
                NotImplementedError("Please check data_format_input option. It has to be tif or npy or fits.")
            
             
            #--------------------------------------
            if len(IMG_A0.shape) == 3: # len(IMG_A0.shape) = IMG_A0.ndim : 배열의 차원 (3차원 (height,width,channels))
                IMG_A0 = IMG_A0.transpose(2, 0 ,1) # 파이토치의 경우 (channels, height, width)이므로 전치시켜줌줌  

            #--------------------------------------
            if self.opt.logscale_input == True: # False
                IMG_A0[np.isnan(IMG_A0)] = 0.1
                IMG_A0[IMG_A0 == 0] = 0.1
                IMG_A0 = np.log10(IMG_A0)
            else:
                IMG_A0[np.isnan(IMG_A0)] = 0
            
            #--------------------------------------[정규화]
            UpIA = np.float(self.opt.saturation_upper_limit_input) # 1.0
            LoIA = np.float(self.opt.saturation_lower_limit_input) # -1.0
            
            if self.opt.saturation_clip_input == True: # False
                label_array = (np.clip(IMG_A0, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            else:
                label_array = (IMG_A0-(UpIA+LoIA)/2)/((UpIA - LoIA)/2) # = IMG_A0 
                
            #--------------------------------------
            label_shape = label_array.shape  #(3,300,275)
            label_array = self.__rotate(label_array)
            label_array = self.__pad(label_array, self.opt.padding_size)
            label_array = self.__random_crop(label_array, label_shape)

            #random_crop
            datasize = trainopt.data_size #data_size = 256
            label_tensor = torch.tensor(label_array, dtype=torch.float32)
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(label_tensor, output_size = (datasize, datasize)) 
            label_tensor = torchvision.transforms.functional.crop(label_tensor, i, j, h, w) # i:이미지 상단에서 시작하는 인덱스(행) , j:이미지 왼쪽에서 시작하는 인덱스(열) -->랜덤

            #random flip
            #random_flip = torchvision.transforms.RandomVerticalFlip.get_parameter(label_tensor)




            #--------------------------------------
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0) #한 차원 늘리기
            
                
# [ Train Target ] ============================================================
            if self.input_format in ["tif", "tiff"]:
                IMG_B0 = np.array(Image.open(self.target_path_list[index]))
            elif self.target_format in ["npy"]:
                IMG_B0 = np.load(self.target_path_list[index], allow_pickle=True)
            elif self.target_format in ["fits", "fts", "fit"]:
                IMG_B0 = np.array(fits.open(self.target_path_list[index])[0].data)
            else:
                NotImplementedError("Please check data_format_target option. It has to be tif or npy or fits.")
            
            #--------------------------------------
            if len(IMG_B0.shape) == 3:
                IMG_B0 = IMG_B0.transpose(2, 0 ,1)
            
            #--------------------------------------
            if self.opt.logscale_target == True:
                IMG_B0[np.isnan(IMG_B0)] = 0.1
                IMG_B0[IMG_B0 == 0] = 0.1
                IMG_B0 = np.log10(IMG_B0)
            else:
                IMG_B0[np.isnan(IMG_B0)] = 0
            
            #--------------------------------------
            IMG_B0[np.isnan(IMG_B0)] = 0
            UpIB = np.float(self.opt.saturation_upper_limit_target)
            LoIB = np.float(self.opt.saturation_lower_limit_target)
            
            if self.opt.saturation_clip_target == True:
                target_array = (np.clip(IMG_B0, LoIB, UpIB)-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            else:
                target_array = (IMG_B0-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            
            #--------------------------------------
            target_shape = target_array.shape
            target_array = self.__rotate(target_array)
            target_array = self.__pad(target_array, self.opt.padding_size)
            target_array = self.__random_crop(target_array, target_shape)

            target_tensor = torch.tensor(target_array, dtype=torch.float32)


            #random crop
            target_tensor = torchvision.transforms.functional.crop(target_tensor, i, j, h, w)

            #--------------------------------------
            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.


# [ Test data ] ===============================================================
        else:
# [ Test Input ] ==============================================================
            if self.input_format in ["tif", "tiff"]:
                IMG_A0 = np.array(Image.open(self.label_path_list[index]))       
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            elif self.input_format in ["fits", "fts", "fit"]:                    
                IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data)
            else:
                NotImplementedError("Please check data_format_input option. It has to be tif or npy or fits.")
            
            #-------------------------------------- 256으로 crop
            IMG_A0 = IMG_A0[self.s:self.s+256, self.e:self.e+256, :]
            
            #--------------------------------------
            if len(IMG_A0.shape) == 3:
                IMG_A0 = IMG_A0.transpose(2, 0 ,1)

            #--------------------------------------
            if self.opt.logscale_input == True:
                IMG_A0[np.isnan(IMG_A0)] = 0.1
                IMG_A0[IMG_A0 == 0] = 0.1
                IMG_A0 = np.log10(IMG_A0)
            else:
                IMG_A0[np.isnan(IMG_A0)] = 0
            
            #--------------------------------------
            UpIA = np.float(self.opt.saturation_upper_limit_input)
            LoIA = np.float(self.opt.saturation_lower_limit_input)
            
            label_array = (np.clip(IMG_A0, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)

            label_tensor = torch.tensor(label_array, dtype=torch.float32)
            
            #--------------------------------------
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
            #--------------------------------------
# [ Test Target ] ==============================================================

            if self.input_format in ["tif", "tiff"]:
                IMG_B0 = np.array(Image.open(self.target_path_list[index]))       
            elif self.target_format in ["npy"]:
                IMG_B0 = np.load(self.target_path_list[index], allow_pickle=True)
            elif self.target_format in ["fits", "fts", "fit"]:                    
                IMG_B0 = np.array(fits.open(self.target_path_list[index])[0].data)
            else:
                NotImplementedError("Please check data_format_input option. It has to be tif or npy or fits.")
            
            #-------------------------------------- 256으로 crop
            IMG_B0 = IMG_B0[self.s:self.s+256, self.e:self.e+256] # (256, 256)
            #print("IMG_B0", IMG_B0.shape) # (256, 256)

            #--------------------------------------
            if len(IMG_B0.shape) == 3:
                IMG_B0 = IMG_B0.transpose(2, 0 ,1)

            #--------------------------------------
            if self.opt.logscale_input == True:
                IMG_B0[np.isnan(IMG_B0)] = 0.1
                IMG_B0[IMG_B0 == 0] = 0.1
                IMG_B0 = np.log10(IMG_B0)
            else:
                IMG_B0[np.isnan(IMG_B0)] = 0
            
            #--------------------------------------
            IMG_B0[np.isnan(IMG_B0)] = 0
            UpIB = np.float(self.opt.saturation_upper_limit_target)
            LoIB = np.float(self.opt.saturation_lower_limit_target)
            #print("IMG_B0", IMG_B0.shape) # (256, 256)
            if self.opt.saturation_clip_target == True:
                target_array = (np.clip(IMG_B0, LoIB, UpIB)-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            else:
                target_array = (IMG_B0-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            #print("target_array", target_array.shape) # (256, 256)
            #--------------------------------------
            target_tensor = torch.tensor(target_array, dtype=torch.float32) # [3, 256, 256]
            #print("target_tensor1", target_tensor.shape)
            #--------------------------------------
            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)

            #print("target_tensor2", target_tensor.shape)

            return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], splitext(split(self.target_path_list[index])[-1])[0]
        
        return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], splitext(split(self.target_path_list[index])[-1])[0]

#------------------------------------------------------------------------------
# [2] Adjust or Measure the Input and Target data sets
                   
    def __random_crop(self, x, array_shape): 
        x = np.array(x)
        if len(x.shape) == 3:
            x = x[:, self.offset_x: self.offset_x + array_shape[1], self.offset_y: self.offset_y +array_shape[2]]
        else:
            x = x[self.offset_x: self.offset_x + array_shape[0], self.offset_y: self.offset_y + array_shape[1]]
        return x



    @staticmethod
    def __pad(x, padding_size):
        if type(padding_size) == int:
            if len(x.shape) == 3:
                padding_size= ((0, 0), (padding_size, padding_size), (padding_size, padding_size))
            else:
                padding_size = ((padding_size, padding_size), (padding_size, padding_size))
        return np.pad(x, pad_width=padding_size, mode="edge")

    def __rotate(self, x):
        return rotate(x, self.angle, mode="nearest", reshape=False)

    @staticmethod
    def __to_numpy(x):
        return np.array(x, dtype=np.float32)

    def __len__(self):
        return len(self.label_path_list)
    
#------------------------------------------------------------------------------

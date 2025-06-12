"""
Test code for the Pix2PixCC Model
@author: Hyun-Jin Jeong (https://jeonghyunjin.com, jeong_hj@khu.ac.kr)
Reference:
1) https://github.com/JeongHyunJin/Pix2PixCC
2) https://arxiv.org/pdf/2204.12068.pdf
"""

from pix2pixCC_Options import TestOption
opt = TestOption().parse()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pix2pixCC_Pipeline import CustomDataset
from pix2pixCC_Networks import Generator
from pix2pixCC_Utils import Manager
import matplotlib.pyplot as plt

# ==============================================================================
# [1] Initial Conditions Setup

if __name__ == '__main__':

    # --------------------------------------------------------------------------
    import os
    import numpy as np
    from glob import glob
    from tqdm import tqdm

    from PIL import Image
    from astropy.io import fits


    # --------------------------------------------------------------------------

    torch.backends.cudnn.benchmark = True

    opt = TestOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda:0')

    STD = opt.dataset_name

    dataset = CustomDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    iters = opt.iteration
    step = opt.save_freq

    # --------------------------------------------------------------------------

    if (iters == False) or (iters == -1):
        #모델 저장되어있는 디렉토리
        dir_model = './checkpoints/{}/Model/*_G.pt'.format(str(STD))
        ITERATIONs = sorted([int(os.path.basename(x).split('_')[0]) for x in glob(dir_model)])

        for ITERATION in ITERATIONs:

            path_model = './checkpoints/{}/Model/{}_G.pt'.format(str(STD), str(ITERATION))
            dir_image_save = './checkpoints/{}/Image/Test/{}'.format(str(STD), str(ITERATION))
            dir_png_save = './checkpoints/{}/Image/Test/{}_png'.format(str(STD), str(ITERATION))


            if os.path.isdir(dir_image_save) == True:
                pass
            else:
                os.makedirs(dir_image_save, exist_ok=True)
                os.makedirs(dir_png_save, exist_ok=True)

                G = torch.nn.DataParallel(Generator(opt)).to(device)
                G.module.load_state_dict(torch.load(path_model))

                manager = Manager(opt)

                with torch.no_grad():
                    G.eval()
                    for input, name in tqdm(test_data_loader):
                        input = input.to(device)

                        fake = G(input)

                        UpIB = opt.saturation_upper_limit_target
                        LoIB = opt.saturation_lower_limit_target

                        np_fake = fake.cpu().numpy().squeeze() * ((UpIB - LoIB) / 2) + (UpIB + LoIB) / 2
                        if opt.saturation_clip_target == True:
                            np_fake = np.clip(np_fake, LoIB, UpIB)

                        # --------------------------------------
                        if len(np_fake.shape) == 3:
                            np_fake = np_fake.transpose(1, 2, 0)

                        # --------------------------------------
                        if opt.logscale_target == True:
                            np_fake = 10 ** (np_fake)

                        # --------------------------------------
                        if opt.data_format_input in ["tif", "tiff", "png", "jpg", "jpeg"]:
                            if opt.data_format_input in ["png", "jpg", "jpeg"]:
                                np_fake = np.asarray(np_fake, np.uint8)
                            pil_image = Image.fromarray(np_fake)
                            pil_image.save(os.path.join(dir_image_save, name[0] + '_AI'))
                        elif opt.data_format_input in ["npy"]:
                            # 테스트 결과 저장 
                            np.save(os.path.join(dir_image_save, name[0] + '_AI'), np_fake, allow_pickle=True)
                            plt.imsave(os.path.join(dir_png_save, name[0] + '_AI.png'), np_fake,vmin =-1, vmax = 1)

                        elif opt.data_format_input in ["fits", "fts"]:
                            fits.writeto(os.path.join(dir_image_save, name[0] + '_AI'), np_fake)
                        else:
                            NotImplementedError(
                                "Please check data_format_target option. It has to be fit or npy or fits.")


    # --------------------------------------------------------------------------

    else:
        # 특정 iter에서 테스트 진행
        ITERATION = int(iters)
        path_model = './checkpoints/{}/Model/{}_G.pt'.format(str(STD), str(ITERATION))
        dir_image_save = './checkpoints/{}/Image/Test/{}'.format(str(STD), str(ITERATION))
        dir_png_save = './checkpoints/{}/Image/Test/{}_png'.format(str(STD), str(ITERATION))

        os.makedirs(dir_image_save, exist_ok=True)
        os.makedirs(dir_png_save, exist_ok=True)

        G = torch.nn.DataParallel(Generator(opt)).to(device)
        G.module.load_state_dict(torch.load(path_model))

        manager = Manager(opt)

        with torch.no_grad():
            G.eval()
            for input, name in tqdm(test_data_loader):
                input = input.to(device)
                fake = G(input)

                UpIB = opt.saturation_upper_limit_target
                LoIB = opt.saturation_lower_limit_target

                np_fake = fake.cpu().numpy().squeeze() * ((UpIB - LoIB) / 2) + (UpIB + LoIB) / 2
                if opt.saturation_clip_target == True:
                    np_fake = np.clip(np_fake, LoIB, UpIB)

                # --------------------------------------
                if len(np_fake.shape) == 3:
                    np_fake = np_fake.transpose(1, 2, 0)

                # --------------------------------------
                if opt.logscale_target == True:
                    np_fake = 10 ** (np_fake)

                if opt.save_scale != 1:
                    np_fake = np_fake * np.float(opt.save_scale)

                # --------------------------------------
                if opt.data_format_input in ["tif", "tiff", "png", "jpg", "jpeg"]:
                    if opt.data_format_input in ["png", "jpg", "jpeg"]:
                        np_fake = np.asarray(np_fake, np.uint8)
                    pil_image = Image.fromarray(np_fake)
                    pil_image.save(os.path.join(dir_image_save, name[0] + '_AI'))
                elif opt.data_format_input in ["npy"]:
                    np.save(os.path.join(dir_image_save, name[0] + '_AI'), np_fake, allow_pickle=True)
                    plt.imsave(os.path.join(dir_png_save, name[0] + '_AI.png'), np_fake,vmin=-1, vmax=1)
                elif opt.data_format_input in ["fits", "fts"]:
                    fits.writeto(os.path.join(dir_image_save, name[0] + '_AI'), np_fake)
                else:
                    NotImplementedError("Please check data_format_target option. It has to be fit or npy or fits.")

# ==============================================================================
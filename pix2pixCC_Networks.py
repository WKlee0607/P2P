"""
Networks of the pix2pixCC model

"""

#==============================================================================

import torch
import torch.nn as nn
from pix2pixCC_Utils import get_grid, get_norm_layer, get_pad_layer
import torch.nn.functional as F
import numpy as np


#==============================================================================
# [1] Generative Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


# 수정-2
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        input_ch = opt.input_ch
        output_ch = opt.target_ch
        n_gf = opt.n_gf
        norm = get_norm_layer(opt.norm_type)
        act = nn.LeakyReLU(0.2, inplace=True)

        self.n_gf = n_gf

        # ----- Encoder -----
        self.d1 = nn.Sequential(nn.Conv2d(input_ch, n_gf, 4, 2, 1), norm(n_gf), act)       # 128
        self.d2 = nn.Sequential(nn.Conv2d(n_gf, 2 * n_gf, 4, 2, 1), norm(2 * n_gf), act)   # 64
        self.d3 = nn.Sequential(nn.Conv2d(2 * n_gf, 4 * n_gf, 4, 2, 1), norm(4 * n_gf), act) # 32
        self.d4 = nn.Sequential(nn.Conv2d(4 * n_gf, 8 * n_gf, 4, 2, 1), norm(8 * n_gf), act) # 16
        self.d5 = nn.Sequential(nn.Conv2d(8 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 8
        self.d6 = nn.Sequential(nn.Conv2d(16 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 4
        self.d7 = nn.Sequential(nn.Conv2d(16 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 2

        # ----- Latent vector -----
        latent_dim = 16 * n_gf
        self.fc_mu = nn.Linear(latent_dim * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2 * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, latent_dim * 2 * 2)

        # ----- Decoder -----
        def upconv(in_ch, out_ch):
            return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), norm(out_ch), act)

        self.u1 = upconv(16 * n_gf, 16 * n_gf)               # 4
        self.u2 = upconv(2 * 16 * n_gf, 16 * n_gf)           # 8
        self.res2 = ResBlock(16 * n_gf)

        self.u3 = upconv(2 * 16 * n_gf, 8 * n_gf)            # 16
        self.u4 = upconv(2 * 8 * n_gf, 4 * n_gf)             # 32
        self.res4 = ResBlock(4 * n_gf)

        self.u5 = upconv(2 * 4 * n_gf, 2 * n_gf)             # 64
        self.u6 = upconv(2 * 2 * n_gf, n_gf)                 # 128
        self.u7 = upconv(2 * n_gf, n_gf)                     # 256
        self.u8 = nn.Conv2d(n_gf, output_ch, 5, 1, 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        e5 = self.d5(e4)
        e6 = self.d6(e5)
        e7 = self.d7(e6)

        # ----- Latent -----
        z_flat = e7.view(e7.size(0), -1)
        mu, logvar = self.fc_mu(z_flat), self.fc_logvar(z_flat)
        z = self.reparameterize(mu, logvar)
        decode = self.fc_decode(z).view(e7.size(0), 16 * self.n_gf, 2, 2)

        # ----- Decoder -----
        d = self.u1(decode)
        d = self.u2(torch.cat([d, e6], dim=1))
        d = self.res2(d)
        d = self.u3(torch.cat([d, e5], dim=1))
        d = self.u4(torch.cat([d, e4], dim=1))
        d = self.res4(d)
        d = self.u5(torch.cat([d, e3], dim=1))
        d = self.u6(torch.cat([d, e2], dim=1))
        d = self.u7(torch.cat([d, e1], dim=1))
        d = self.u8(d)

        return torch.tanh(d), mu, logvar  # 또는 clamp(d, -1, 1)

# 수정-1
'''
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        input_ch = opt.input_ch
        output_ch = opt.target_ch
        n_gf = opt.n_gf
        norm = get_norm_layer(opt.norm_type)
        act = Mish()

        self.n_gf = opt.n_gf
        
        # ----- Encoder (Downsampling) -----
        self.d_layer1 = nn.Sequential(nn.Conv2d(input_ch, n_gf, 4, 2, 1), norm(n_gf), act)   # 128
        self.d_layer2 = nn.Sequential(nn.Conv2d(n_gf, 2 * n_gf, 4, 2, 1), norm(2 * n_gf), act)   # 64
        self.d_layer3 = nn.Sequential(nn.Conv2d(2 * n_gf, 4 * n_gf, 4, 2, 1), norm(4 * n_gf), act) # 32
        self.d_layer4 = nn.Sequential(nn.Conv2d(4 * n_gf, 8 * n_gf, 4, 2, 1), norm(8 * n_gf), act) # 16
        self.d_layer5 = nn.Sequential(nn.Conv2d(8 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 8
        self.d_layer6 = nn.Sequential(nn.Conv2d(16 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 4
        self.d_layer7 = nn.Sequential(nn.Conv2d(16 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 2

        # ----- Latent vector z -----
        latent_dim = n_gf * 16 # 1024
        self.fc_mu = nn.Linear(16 * n_gf * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(16 * n_gf * 2 * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 16 * n_gf * 2 * 2)

        # ----- Decoder (Upsampling) -----
        self.u_layer1 = nn.Sequential(nn.ConvTranspose2d(16 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 4
        self.u_layer2 = nn.Sequential(nn.ConvTranspose2d(2 * 16 * n_gf, 16 * n_gf, 4, 2, 1), norm(16 * n_gf), act) # 8
        self.u_layer3 = nn.Sequential(nn.ConvTranspose2d(2 * 16 * n_gf, 8 * n_gf, 4, 2, 1), norm(8 * n_gf), act) # 16
        self.u_layer4 = nn.Sequential(nn.ConvTranspose2d(2 * 8 * n_gf, 4 * n_gf, 4, 2, 1), norm(4 * n_gf), act) # 32
        self.u_layer5 = nn.Sequential(nn.ConvTranspose2d(2 * 4 * n_gf, 2 * n_gf, 4, 2, 1), norm(2 * n_gf), act) # 64
        self.u_layer6 = nn.Sequential(nn.ConvTranspose2d(2 * 2 * n_gf, n_gf, 4, 2, 1), norm(n_gf), act) # 128
        self.u_layer7 = nn.Sequential(nn.ConvTranspose2d(n_gf, n_gf, 4, 2, 1), norm(n_gf), act) # 256
        self.u_layer8 = nn.Conv2d(n_gf, output_ch, 5, 1, 2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # ----- Encoder -----
        out1 = self.d_layer1(x)
        out2 = self.d_layer2(out1)
        out3 = self.d_layer3(out2)
        out4 = self.d_layer4(out3)
        out5 = self.d_layer5(out4)
        out6 = self.d_layer6(out5)
        out7 = self.d_layer7(out6)  # shape: [B, 16 * n_gf, 2, 2]

        flat = out7.view(out7.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)
        
        # ----- Decoder -----
        decode = self.fc_decode(z).view(out7.size(0), 16 * self.n_gf, 2, 2)

        out = self.u_layer1(decode)
        out = self.u_layer2(torch.cat((out, out6), dim=1))
        out = self.u_layer3(torch.cat((out, out5), dim=1))
        out = self.u_layer4(torch.cat((out, out4), dim=1))
        out = self.u_layer5(torch.cat((out, out3), dim=1))
        out = self.u_layer6(torch.cat((out, out2), dim=1))
        out = self.u_layer7(out)
        out = self.u_layer8(out)

        return out, mu, logvar

'''


'''
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        #----------------------------------------------------------------------        
        input_ch = opt.input_ch # 인풋 3채널 
        output_ch = opt.target_ch # 타겟 1채널 
        n_gf = opt.n_gf # 출력 채널 수 64
        norm = get_norm_layer(opt.norm_type) #instanceNorm2d
        act = Mish()
        pad = get_pad_layer(opt.padding_type) #  layer = nn.ReflectionPad2d
        trans_conv = opt.trans_conv # transposed convolution 사용 

        #---------------------------------------------------------------------- Down_sample
        self.d_layer1 = nn.Sequential(
            nn.Conv2d(input_ch, n_gf, kernel_size=4, stride=2, padding=1), norm(n_gf), act # 128
        ) 
        self.d_layer2 = nn.Sequential(
            nn.Conv2d(n_gf, 2 * n_gf, kernel_size=4, stride=2, padding=1), norm(2 * n_gf), act, # 64
        )
        self.d_layer3 = nn.Sequential(
            nn.Conv2d(2*n_gf, 4 * n_gf,kernel_size=4, stride=2, padding=1), norm(4 * n_gf), act, # 32, 4
        )
        self.d_layer4 = nn.Sequential(
            nn.Conv2d(4*n_gf, 8 * n_gf, kernel_size=4, stride=2, padding=1), norm(8 * n_gf), act, # 16
        )
        self.d_layer5 = nn.Sequential(
            nn.Conv2d(8*n_gf, 16 * n_gf, kernel_size=4, stride=2, padding=1), norm(16 * n_gf), act # 8
        )
        self.d_layer6 = nn.Sequential(
            nn.Conv2d(16*n_gf, 16 * n_gf, kernel_size=4, stride=2, padding=1), norm(16 * n_gf), act # 4
        )
        self.d_layer7 = nn.Sequential(
            nn.Conv2d(16*n_gf, 16 * n_gf, kernel_size=4, stride=2, padding=1), norm(16 * n_gf), act # 2
        )
        #---------------------------------------------------------------------- Atten

        #---------------------------------------------------------------------- Up_sample
        self.u_layer1 = nn.Sequential(
            nn.ConvTranspose2d(16 * n_gf, n_gf * 16, kernel_size=4, stride=2, padding=1), norm(n_gf * 16), act # 4
        )
        self.u_layer2 = nn.Sequential(
            nn.ConvTranspose2d(2 * 16 * n_gf, n_gf * 16, kernel_size=4, stride=2, padding=1), norm(n_gf * 16), act # 8
        )
        self.u_layer3 = nn.Sequential(
            nn.ConvTranspose2d(2 * 16 * n_gf, n_gf * 8, kernel_size=4, stride=2, padding=1), norm(n_gf * 8), act # 16
        )
        self.u_layer4 = nn.Sequential(
            nn.ConvTranspose2d(2 * 8 * n_gf, n_gf * 4, kernel_size=4, stride=2, padding=1), norm(n_gf * 4), act # 32
        )
        self.u_layer5 = nn.Sequential(
            nn.ConvTranspose2d(2 * 4 * n_gf, n_gf * 2, kernel_size=4, stride=2, padding=1), norm(n_gf * 2), act # 64
        )
        self.u_layer6 = nn.Sequential(
            nn.ConvTranspose2d(2 * 2 * n_gf, n_gf, kernel_size=4, stride=2, padding=1), norm(n_gf), act # 128
        )
        self.u_layer7 = nn.Sequential(
            nn.ConvTranspose2d(n_gf, n_gf, kernel_size=4, stride=2, padding=1), norm(n_gf), act # 256
        )
        self.u_layer8 = nn.Sequential(
            nn.Conv2d(n_gf, output_ch, kernel_size=5, stride=1, padding=2), # 256
        )

        self.last = nn.Softsign() 
        #----------------------------------------------------------------------
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        # down
        out1 = self.d_layer1(x) # 128, n_gf
        out2 = self.d_layer2(out1) # 64, 2
        out3 = self.d_layer3(out2) # 32, 4
        out4 = self.d_layer4(out3) # 16, 8
        out5 = self.d_layer5(out4) # 8, 16
        out6 = self.d_layer6(out5) # 4, 16
        out7 = self.d_layer7(out6) # 2, 16

        # up
        out = self.u_layer1(out7) # 4, 16
        out = self.u_layer2(torch.cat((out, out6), dim=1)) # 8, 16
        out = self.u_layer3(torch.cat((out, out5), dim=1)) # 16, 8
        out = self.u_layer4(torch.cat((out, out4), dim=1)) # 32, 4
        out = self.u_layer5(torch.cat((out, out3), dim=1)) # 64, 2
        out = self.u_layer6(torch.cat((out, out2), dim=1)) # 128, 1
        out = self.u_layer7(out) # 256, 1
        out = self.u_layer8(out)
        return self.last(out)
'''


#------------------------------------------------------------------------------

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



#==============================================================================
# [2] Discriminative Network
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()

        if opt.ch_balance > 0:
            ch_ratio = float(opt.input_ch) / float(opt.target_ch) * opt.ch_balance
            if ch_ratio > 1:
                input_channel = opt.input_ch + opt.target_ch * int(ch_ratio)
            elif ch_ratio < 1:
                input_channel = opt.input_ch * int(1 / ch_ratio) + opt.target_ch
            else:
                input_channel = opt.input_ch + opt.target_ch
        else:
            input_channel = opt.input_ch + opt.target_ch

        ndf = opt.n_df
        act = nn.LeakyReLU(0.2, inplace=True)
        norm_type = getattr(opt, 'norm_type', 'instance')
        use_spectral = getattr(opt, 'use_spectral', True)

        def conv(in_c, out_c, stride=2, dilation=1, norm=True):
            padding = dilation  # 유지된 출력 크기를 위해 dilation만큼 패딩
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=padding, dilation=dilation)]
            if use_spectral:
                layers[0] = nn.utils.spectral_norm(layers[0])
            if norm:
                layers.append(get_norm_layer(norm_type)(out_c))
            layers.append(act)
            return nn.Sequential(*layers)

        # ✅ Receptive field 강화된 구조
        self.block1 = conv(input_channel, ndf, stride=2)
        self.block2 = conv(ndf, ndf * 2, stride=2)
        self.block3 = conv(ndf * 2, ndf * 4, stride=2)
        self.block4 = conv(ndf * 4, ndf * 8, stride=1, dilation=2)  # dilated conv
        self.block5 = conv(ndf * 8, ndf * 8, stride=1, dilation=4)  # dilated conv
        self.final = nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)
        if use_spectral:
            self.final = nn.utils.spectral_norm(self.final)

    def forward(self, x):
        feat = []
        x = self.block1(x); feat.append(x)
        x = self.block2(x); feat.append(x)
        x = self.block3(x); feat.append(x)
        x = self.block4(x); feat.append(x)
        x = self.block5(x); feat.append(x)
        out = self.final(x)
        feat.append(out)
        return feat



# 원본
"""
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        
        #----------------------------------------------------------------------
        if opt.ch_balance > 0:
            ch_ratio = np.float(opt.input_ch)/np.float(opt.target_ch)
            ch_ratio *= opt.ch_balance
            if ch_ratio > 1:
                input_channel = opt.input_ch + opt.target_ch*np.int(ch_ratio)                            
            elif ch_ratio < 1:
                input_channel = opt.input_ch*np.int(1/ch_ratio) + opt.target_ch
            else:
                input_channel = opt.input_ch + opt.target_ch
        else:
            input_channel = opt.input_ch + opt.target_ch
        
        #----------------------------------------------------------------------
        act = nn.LeakyReLU(0.2, inplace=True)
        n_df = opt.n_df #64
        norm = nn.InstanceNorm2d
        
        #----------------------------------------------------------------------
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=2), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))
            
        #----------------------------------------------------------------------
        
        
    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input
"""

"""
# 수정본
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        
        #----------------------------------------------------------------------
        if opt.ch_balance > 0:
            ch_ratio = np.float(opt.input_ch)/np.float(opt.target_ch)
            ch_ratio *= opt.ch_balance
            if ch_ratio > 1:
                input_channel = opt.input_ch + opt.target_ch*np.int(ch_ratio)                            
            elif ch_ratio < 1:
                input_channel = opt.input_ch*np.int(1/ch_ratio) + opt.target_ch
            else:
                input_channel = opt.input_ch + opt.target_ch
        else:
            input_channel = opt.input_ch + opt.target_ch
        
        #----------------------------------------------------------------------
        act = nn.LeakyReLU(0.2, inplace=True)
        n_df = opt.n_df #64
        norm = nn.InstanceNorm2d
        
        #----------------------------------------------------------------------
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]] # 32
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=2), norm(8 * n_df), act]] # 16
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]] 

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))
            
        #----------------------------------------------------------------------
        
        
    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input
"""
#------------------------------------------------------------------------------
# 수정본
class Discriminator(nn.Module):
    def __init__(self, opt, use_gaussian_blur=False):
        super(Discriminator, self).__init__()

        self.n_D = opt.n_D
        self.use_blur_input = getattr(opt, 'use_blur_input', False)
        self.use_gaussian_blur = getattr(opt, 'use_gaussian_blur', False)

        # Discriminator for each scale
        self.scales = nn.ModuleList([PatchDiscriminator(opt) for _ in range(self.n_D)])

        # Downsampling module between scales
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.use_gaussian_blur = use_gaussian_blur

        # Optional Gaussian blur for anti-artifact support
        if self.use_gaussian_blur:
            self.blur = GaussianBlur(kernel_size=5, sigma=(1.0, 1.0))

        print("Total D parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        outputs = []
        for i, D in enumerate(self.scales):
            x_input = x

            # Optional: Concatenate blurred version to enhance salt-and-pepper detection
            if self.use_blur_input and i == 0:
                blurred = self.blur(x) if self.use_gaussian_blur else nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
                x_input = torch.cat([x, blurred], dim=1)

            outputs.append(D(x_input))

            if i != self.n_D - 1:
                x = self.downsample(x)

        return outputs  # List of feature lists from each scale

# 원본
"""
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        #----------------------------------------------------------------------
        for i in range(opt.n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(opt))
        self.n_D = opt.n_D #1

        #----------------------------------------------------------------------
        #print(self)
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
                
        return result
"""


#==============================================================================
# [3] Objective (Loss) functions

class Loss(object):
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device #torch.device('cuda:0' if opt.gpu_ids != -1 else 'cpu:0')
        self.dtype = torch.float16 if opt.data_type == 16 else torch.float32
        
        self.MSE = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = opt.n_D


    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0
        
        fake, mu, logvar = G(input)
        
        
        #----------------------------------------------------------------------
        # [3-1] Get Real and Fake (Generated) pairs and features 
        
        if self.opt.ch_balance > 0:        
            
            real_pair = torch.cat((input, target), dim=1)
            fake_pair = torch.cat((input, fake.detach()), dim=1)
            
            ch_plus = 0
            ch_ratio = np.float(self.opt.input_ch)/np.float(self.opt.target_ch)
            ch_ratio *= self.opt.ch_balance
            if ch_ratio > 1:
                for dr in range(np.int(ch_ratio)-1):
                    real_pair = torch.cat((real_pair, target), dim=1)
                    fake_pair = torch.cat((fake_pair, fake.detach()), dim=1)
                    ch_plus += self.opt.target_ch                         
            
            elif ch_ratio < 1:                
                for _ in range(np.int(1/ch_ratio)-1):
                    real_pair = torch.cat((input, real_pair), dim=1)
                    fake_pair = torch.cat((input, fake_pair), dim=1)
                    ch_plus += self.opt.input_ch
                
            else:
                pass
            
            real_features = D(real_pair)
            fake_features = D(fake_pair)
        else:
            real_features = D(torch.cat((input, target), dim=1))
            fake_features = D(torch.cat((input, fake.detach()), dim=1))
        
        
        #----------------------------------------------------------------------
        # [3-2] Compute LSGAN loss for the discriminator
        
        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device, self.dtype)

            loss_D += (self.MSE(real_features[i][-1], real_grid) +
                    self.MSE(fake_features[i][-1], fake_grid)) * 0.5
        
        
        #----------------------------------------------------------------------
        # [3-3] Compute LSGAN loss and Feature Matching loss for the generator
        
        if self.opt.ch_balance > 0:  
            fake_pair = torch.cat((input, fake), dim=1)
            
            if ch_ratio > 1:
                for _ in range(np.int(ch_ratio)-1):
                    fake_pair = torch.cat((fake_pair, fake), dim=1)
            elif ch_ratio < 1:
                for _ in range(np.int(1/ch_ratio)-1):
                    fake_pair = torch.cat((input, fake_pair), dim=1)
            else:
                pass
            
            fake_features = D(fake_pair)
        else:
            fake_features = D(torch.cat((input, fake), dim=1))
            
        
        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.MSE(fake_features[i][-1], real_grid) * 0.5 * self.opt.lambda_LSGAN
            
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())
                
            loss_G += loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM
        
        
        #----------------------------------------------------------------------
        # [3-4] Compute Correlation Coefficient loss for the generator
        
        for i in range(self.opt.n_CC):
            real_down = target.to(self.device, self.dtype)
            fake_down = fake.to(self.device, self.dtype)
            for _ in range(i):
                real_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(real_down)
                fake_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(fake_down)
            
            loss_CC = self.__Inspector(real_down, fake_down)
            loss_G += loss_CC * (1.0 / self.opt.n_CC) * self.opt.lambda_CC
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / fake.size(0)
        loss_G += kl_loss
        #----------------------------------------------------------------------
        return loss_D, loss_G, target, fake
        
    
#==============================================================================
# [4] Inspector
    
    def __Inspector(self, target, fake):
                
        rd = target - torch.nanmean(target)
        fd = fake - torch.nanmean(fake)
        
        r_num = torch.nansum(rd * fd)
        r_den = torch.sqrt(torch.nansum(rd ** 2)) * torch.sqrt(torch.nansum(fd ** 2))
        PCC_val = r_num/(r_den + self.opt.eps)
        
        #----------------------------------------------------------------------
        if self.opt.ccc == True:
            numerator = 2*PCC_val*torch.std(target)*torch.std(fake)
            denominator = (torch.var(target) + torch.var(fake)
                           + (torch.nanmean(target) - torch.nanmean(fake))**2)
            
            CCC_val = numerator/(denominator + self.opt.eps)
            loss_CC = (1.0 - CCC_val)
        
        else:
            loss_CC = (1.0 - PCC_val)
            
        #----------------------------------------------------------------------
        return loss_CC
    
    
#==============================================================================







""" # 원본
#==============================================================================
# [1] Generative Network

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        #----------------------------------------------------------------------        
        input_ch = opt.input_ch # 인풋 3채널 
        output_ch = opt.target_ch # 타겟 1채널 
        n_gf = opt.n_gf # 출력 채널 수 64
        norm = get_norm_layer(opt.norm_type) #instanceNorm2d
        act = Mish()
        pad = get_pad_layer(opt.padding_type) #  layer = nn.ReflectionPad2d
        trans_conv = opt.trans_conv # transposed convolution 사용 

        #----------------------------------------------------------------------
        model = [] # 모델 리스트 

        # model 리스트에 첫번째 레이어 추가
        model += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0), norm(n_gf), act]
        # --> [nn.ReflectionPad2d(3) , nn.Conv2d(3, 64, kernel_size=7, padding=0), nn.InstanceNorm2d(64) 정규화 , Mish() 활성화 함수] 
        
        # 다운샘플링
        for _ in range(opt.n_downsample): # n_downsample:4
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=5, padding=2, stride=2), norm(2 * n_gf), act]
            n_gf *= 2


        for _ in range(opt.n_residual): #n_residual:9
            model += [ResidualBlock(n_gf, pad, norm, act)]

        # 업샘플링 
        for n_up in range(opt.n_downsample):
            #------------------------------------------------------------------
            if trans_conv == True:
                model += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                          norm(n_gf//2), act]
            else:
                model += [nn.UpsamplingBilinear2d(scale_factor=2)]
                model += [pad(1), nn.Conv2d(n_gf, n_gf//2, kernel_size=3, padding=0, stride=1), norm(n_gf//2), act]
            #------------------------------------------------------------------                        
            n_gf //= 2
        
        
        model += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)
        #----------------------------------------------------------------------
        
        #print(self)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def forward(self, x):
        return self.model(x)

#------------------------------------------------------------------------------
        
class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


#------------------------------------------------------------------------------

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



#==============================================================================
# [2] Discriminative Network

class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        
        #----------------------------------------------------------------------
        if opt.ch_balance > 0:
            ch_ratio = np.float(opt.input_ch)/np.float(opt.target_ch)
            ch_ratio *= opt.ch_balance
            if ch_ratio > 1:
                input_channel = opt.input_ch + opt.target_ch*np.int(ch_ratio)                            
            elif ch_ratio < 1:
                input_channel = opt.input_ch*np.int(1/ch_ratio) + opt.target_ch
            else:
                input_channel = opt.input_ch + opt.target_ch
        else:
            input_channel = opt.input_ch + opt.target_ch
        
        #----------------------------------------------------------------------
        act = nn.LeakyReLU(0.2, inplace=True)
        n_df = opt.n_df #64
        norm = nn.InstanceNorm2d
        
        #----------------------------------------------------------------------
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))
            
        #----------------------------------------------------------------------
        
        
    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input

#------------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        #----------------------------------------------------------------------
        for i in range(opt.n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(opt))
        self.n_D = opt.n_D #1

        #----------------------------------------------------------------------
        #print(self)
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
                
        return result



#==============================================================================
# [3] Objective (Loss) functions

class Loss(object):
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device #torch.device('cuda:0' if opt.gpu_ids != -1 else 'cpu:0')
        self.dtype = torch.float16 if opt.data_type == 16 else torch.float32
        
        self.MSE = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = opt.n_D


    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0
        
        fake = G(input)
        
        
        #----------------------------------------------------------------------
        # [3-1] Get Real and Fake (Generated) pairs and features 
        
        if self.opt.ch_balance > 0:        
            
            real_pair = torch.cat((input, target), dim=1)
            fake_pair = torch.cat((input, fake.detach()), dim=1)
            
            ch_plus = 0
            ch_ratio = np.float(self.opt.input_ch)/np.float(self.opt.target_ch)
            ch_ratio *= self.opt.ch_balance
            if ch_ratio > 1:
                for dr in range(np.int(ch_ratio)-1):
                    real_pair = torch.cat((real_pair, target), dim=1)
                    fake_pair = torch.cat((fake_pair, fake.detach()), dim=1)
                    ch_plus += self.opt.target_ch                         
            
            elif ch_ratio < 1:                
                for _ in range(np.int(1/ch_ratio)-1):
                    real_pair = torch.cat((input, real_pair), dim=1)
                    fake_pair = torch.cat((input, fake_pair), dim=1)
                    ch_plus += self.opt.input_ch
                
            else:
                pass
            
            real_features = D(real_pair)
            fake_features = D(fake_pair)
        else:
            real_features = D(torch.cat((input, target), dim=1))
            fake_features = D(torch.cat((input, fake.detach()), dim=1))
        
        
        #----------------------------------------------------------------------
        # [3-2] Compute LSGAN loss for the discriminator
        
        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device, self.dtype)

            loss_D += (self.MSE(real_features[i][-1], real_grid) +
                    self.MSE(fake_features[i][-1], fake_grid)) * 0.5
        
        
        #----------------------------------------------------------------------
        # [3-3] Compute LSGAN loss and Feature Matching loss for the generator
        
        if self.opt.ch_balance > 0:  
            fake_pair = torch.cat((input, fake), dim=1)
            
            if ch_ratio > 1:
                for _ in range(np.int(ch_ratio)-1):
                    fake_pair = torch.cat((fake_pair, fake), dim=1)
            elif ch_ratio < 1:
                for _ in range(np.int(1/ch_ratio)-1):
                    fake_pair = torch.cat((input, fake_pair), dim=1)
            else:
                pass
            
            fake_features = D(fake_pair)
        else:
            fake_features = D(torch.cat((input, fake), dim=1))
            
        
        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.MSE(fake_features[i][-1], real_grid) * 0.5 * self.opt.lambda_LSGAN
            
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())
                
            loss_G += loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM
        
        
        #----------------------------------------------------------------------
        # [3-4] Compute Correlation Coefficient loss for the generator
        
        for i in range(self.opt.n_CC):
            real_down = target.to(self.device, self.dtype)
            fake_down = fake.to(self.device, self.dtype)
            for _ in range(i):
                real_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(real_down)
                fake_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(fake_down)
            
            loss_CC = self.__Inspector(real_down, fake_down)
            loss_G += loss_CC * (1.0 / self.opt.n_CC) * self.opt.lambda_CC
        
        #----------------------------------------------------------------------
        return loss_D, loss_G, target, fake
        
        
       
    
    
    
#==============================================================================
# [4] Inspector
    
    def __Inspector(self, target, fake):
                
        rd = target - torch.nanmean(target)
        fd = fake - torch.nanmean(fake)
        
        r_num = torch.nansum(rd * fd)
        r_den = torch.sqrt(torch.nansum(rd ** 2)) * torch.sqrt(torch.nansum(fd ** 2))
        PCC_val = r_num/(r_den + self.opt.eps)
        
        #----------------------------------------------------------------------
        if self.opt.ccc == True:
            numerator = 2*PCC_val*torch.std(target)*torch.std(fake)
            denominator = (torch.var(target) + torch.var(fake)
                           + (torch.nanmean(target) - torch.nanmean(fake))**2)
            
            CCC_val = numerator/(denominator + self.opt.eps)
            loss_CC = (1.0 - CCC_val)
        
        else:
            loss_CC = (1.0 - PCC_val)
            
        #----------------------------------------------------------------------
        return loss_CC
    
    
#==============================================================================
"""
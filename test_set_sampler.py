
import os
import torch
import lpips
from PIL import Image
import torchvision.transforms as T
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = lpips.LPIPS(net='vgg').to(device)

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
    ])
    img_t = transform(img).unsqueeze(0).to(device)
    return img_t

folder_A = '/data/fovert/repos/repos/P2PCC/Final_set/real'
folder_B = '/data/fovert/repos/repos/P2PCC/Final_set/Ours_90_half' # FID: 113.0615, LPIPS: 0.2901
#folder_B = '/data/fovert/repos/repos/P2PCC/Final_set/Ori_90' # FID: 185.5535, LPIPS: 0.3684

lpips_scores = []

A = sorted(os.listdir(folder_A))
B = sorted(os.listdir(folder_B))

for i in range(len(A)):
    path_real = os.path.join(folder_A, A[i])
    path_fake = os.path.join(folder_B, B[i])

    if os.path.isfile(path_fake):
        img_real = preprocess_image(path_real)
        img_fake = preprocess_image(path_fake)

        with torch.no_grad():
            dist = loss_fn(img_real, img_fake).item()
        lpips_scores.append(dist)
        print(f'{A[i]}: LPIPS = {dist:.4f}')

if lpips_scores:
    print(f'Average LPIPS: {np.mean(lpips_scores):.4f}')
else:
    print('No matching files found.')


'''
import os
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
import numpy as np

folder_A = '/data/fovert/repos/repos/P2PCC/Final_set/real'
folder_B = '/data/fovert/repos/repos/P2PCC/Final_set/Ours_90_half' # FID: 113.0615(down), LPIPS: 0.2901(down), SSIM: 0.6995(up)
#folder_B = '/data/fovert/repos/repos/P2PCC/Final_set/Ori_90' # FID: 185.5535(down), LPIPS: 0.3684(down), SSIM: 0.5248(up)
ssim_values = []
count = 0

A = sorted(os.listdir(folder_A))
B = sorted(os.listdir(folder_B))

for i in range(len(A)):
#for filename in os.listdir(folder_A):
    path_A = os.path.join(folder_A, A[i])
    path_B = os.path.join(folder_B, B[i])

    #if os.path.isfile(path_B):  # 같은 파일명이 있으면
    img_A = imread(path_A)
    img_B = imread(path_B)

    # RGB일 때 multichannel=True로 SSIM 계산
    score = ssim(img_A, img_B, data_range=img_B.max() - img_B.min(), channel_axis=-1)

    ssim_values.append(score)
    count += 1
    print(f'{A[i]}: SSIM = {score:.4f}')

# 전체 평균 SSIM
if count > 0:
    avg_ssim = np.mean(ssim_values)
    print(f'\n✅ 전체 평균 SSIM: {avg_ssim:.4f}')
else:
    print('❌ 동일 파일명이 없습니다.')
'''

'''
import os
import shutil

# 폴더 경로
folder_A = '/data/fovert/repos/repos/P2PCC/Final_set/Ori_90'  # 원래 기준 파일이 있는 폴더
folder_B = '/data/fovert/repos/repos/P2PCC/Final_set/Ours_90'  # 동일 파일명을 찾을 폴더
folder_C = '/data/fovert/repos/repos/P2PCC/Final_set/Ours_90_half'  # 저장할 폴더

# C 폴더가 없으면 생성
os.makedirs(folder_C, exist_ok=True)

# A 폴더의 파일명을 기준으로 B 폴더에서 동일 파일 찾아서 C로 복사
for filename in os.listdir(folder_A):
    #src_file = os.path.join(folder_B, filename)
    #if os.path.isfile(src_file):  # 동일 이름 파일이 있는 경우
    #new_filename = filename.replace('Check_', 'Check_61', 1)
    new_filename = filename
    src_file = os.path.join(folder_B, new_filename)
    dest_file = os.path.join(folder_C, new_filename)
    shutil.copy2(src_file, dest_file)  # 원래 메타데이터도 함께 복사
    print(f"✅ 복사 성공: {new_filename}")

print("💡 작업 완료!")
'''

'''
# FID 계산
from pytorch_fid import fid_score
import torch
#path_real = '/data/fovert/repos/repos/P2PCC/Final_set/11/real'
#path_fake = '/data/fovert/repos/repos/P2PCC/Final_set/11/fake' # FID: 113.0615

path_real = '/data/fovert/repos/repos/P2PCC/Final_set/real'
path_fake = '/data/fovert/repos/repos/P2PCC/Final_set/Ours_90_half' # FID: 74.6402
#path_fake = '/data/fovert/repos/repos/P2PCC/Final_set/Ori_90' # FID: 185.5535

fid_value = fid_score.calculate_fid_given_paths(
    [path_real, path_fake],
    batch_size=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dims=2048
)

print(f'FID: {fid_value:.4f}')

'''


'''
# 크기 확인
import numpy as np
import matplotlib.pyplot as plt
import os

save_path = '/data/fovert/repos/repos/P2PCC/Images/oo/'

input_path_t = "/local_datasets/SAT2Rain/daytime_model/test/input/VI006_IR105_WV063_202108010000_ra.npy"
output_paht_t = "/local_datasets/SAT2Rain/daytime_model/test/target/CMX_202108010900_ra.npy"

i = np.load(input_path_t) # (720, 576, 3) - [H, W, C]
o = np.load(output_paht_t) # (720, 576) - [H, W]
s, e = 290, 200

i = i[s:s+256, e:e+256, :]
o = o[s:s+256, e:e+256]


print(i.shape, o.shape)
plt.figure(figsize=(6, 6))
plt.imshow(i, cmap='brg')
plt.savefig(f'{save_path}/i.png', bbox_inches='tight', pad_inches=0.1)
plt.close()


plt.figure(figsize=(6, 6))
plt.imshow(o, cmap='gray')
plt.savefig(f'{save_path}/o.png', bbox_inches='tight', pad_inches=0.1)
plt.close()

'''


'''
import numpy as np

input_path = "/local_datasets/SAT2Rain/daytime_model/train/input/VI006_IR105_WV063_202002280710_ra.npy"
output_paht = "/local_datasets/SAT2Rain/daytime_model/train/target/CMX_202006251410_ra.npy"

input_path_t = "/local_datasets/SAT2Rain/daytime_model/test/input/VI006_IR105_WV063_202108310800_ra.npy"
output_paht_t = "/local_datasets/SAT2Rain/daytime_model/test/target/CMX_202108311700_ra.npy"

i = np.load(input_path) # (300, 275, 3) - [H, W, C]
o = np.load(output_paht) # (300, 275) - [H, W]

i_t = np.load(input_path_t) # (720, 576, 3) - [H, W, C]
o_t = np.load(output_paht_t) # (720, 576) - [H, W]

print(i.shape, o.shape)
print(i_t.shape, o_t.shape)

#print(i.max(), i.min()) # 0.5 , -0.7 -> 얘는 이미 [1, -1]인듯
#print(o.max(), o.min()) # 0.48, -1.03
import numpy as np
import matplotlib.pyplot as plt
import os

name = "t_input"

# 1️⃣ npy 파일 불러오
s, e = 290, 200
npy_path = f'/data/fovert/repos/repos/P2PCC/Images/{name}.npy' 
data = np.load(npy_path)[s:s+256, e:e+256, :]
print(data.shape)
#data = np.transpose(data, (1, 2, 0))

# 2️⃣ 데이터 형태 출력
print('데이터 shape:', data.shape)

# 3️⃣ 저장할 경로 지정
save_path = f'/data/fovert/repos/repos/P2PCC/Images/{name}.png'  

# 4️⃣ 시각화 및 저장
plt.figure(figsize=(6, 6))
plt.imshow(data, cmap='gray')

# 저장
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
print(f'이미지가 {save_path}에 저장되었습니다.')

#plt.show()
'''
'''

import numpy as np
import matplotlib.pyplot as plt
import os

path = "/data/fovert/repos/repos/P2PCC/target_imgs"
files = os.listdir(path)
s, e = 290, 200
cnt = 0
for f in files:
    data = np.load(os.path.join(path, f))
    print(cnt)
    cnt += 1
    data = data[s:s+256, e:e+256]
    save_path = f'/data/fovert/repos/repos/P2PCC/Images/target_imgs/{f[:-4]}.png' 

    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray')

    # 저장
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

'''
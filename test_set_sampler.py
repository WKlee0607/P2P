# 크기 확인
'''

import numpy as np

input_path = "/local_datasets/SAT2Rain/daytime_model/train/input/VI006_IR105_WV063_202002280710_ra.npy"
output_paht = "/local_datasets/SAT2Rain/daytime_model/train/target/CMX_202006251410_ra.npy"

input_path_t = "/local_datasets/SAT2Rain/daytime_model/test/input/VI006_IR105_WV063_202108120800_ra.npy"
output_paht_t = "/local_datasets/SAT2Rain/daytime_model/test/target/CMX_202108081300_ra.npy"

i = np.load(input_path) # (300, 275, 3) - [H, W, C]
o = np.load(output_paht) # (300, 275) - [H, W]

i_t = np.load(input_path_t) # (720, 576, 3) - [H, W, C]
o_t = np.load(output_paht_t) # (720, 576) - [H, W]

print(i.shape, o.shape)
print(i_t.shape, o_t.shape)

#print(i.max(), i.min()) # 0.5 , -0.7 -> 얘는 이미 [1, -1]인듯
#print(o.max(), o.min()) # 0.48, -1.03
'''
'''
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


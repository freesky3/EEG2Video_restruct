import numpy as np
file = np.load(r"D:\sjtu文件夹\PLUS课程文件夹\PRP\DATA\watching_data\PSD_DE\chenjiaxin_20250630_session1__PSD_DE.npy")
print(file.shape)

from einops import rearrange
data = rearrange(file, 'a b c d e -> b (a c) d e')
print(data.shape)

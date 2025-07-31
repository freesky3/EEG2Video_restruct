'''
input: wating_data or imaging_data from video_division.py
output: DE_list and PSD_list for each electrode
'''

from EEG_preprocessing.DE_PSD import DE_PSD
import numpy as np

file_path = r"D:\sjtu文件夹\PLUS课程文件夹\PRP\DATA\watching_data\chenjiaxin_20250630_session1.npy"
file = np.load(file_path)

DE_list = []
PSD_list = []

for i in range(0,file.shape[0]):
    DE_list_i = []
    PSD_list_i = []
    for j in range(0, file.shape[1]):
        DE_list_j = []
        PSD_list_j = []
        DE, PSD = DE_PSD(file[i,j,:], fre=200, time_win=2)
        DE_list_i.append(DE)
        PSD_list_i.append(PSD)
    DE_list.append(np.stack(DE_list_i, 0))
    PSD_list.append(np.stack(PSD_list_i, 0))
DE_list = np.stack(DE_list, 0)
PSD_list = np.stack(PSD_list, 0)
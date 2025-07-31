'''
compute the differential entropy(DE) and power spectral density(PSD) of EEG data from video_division.py
'''

import numpy as np
import os
from DE_PSD import DE_PSD
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv("variable.env")

def DE_PSD_clip(data, time_win):
    '''
    Args:
    data: EEG signal, (5, 62, 505*200)
        5: 5 videos, 62: 62 electrodes, 505*200: 505 seconds per video * 200 points per second
    Returns:
        DE_list: differential entropy, (5, 50, 62, 5)
            5: 5 videos, 50: 50 clips per video, 62: 62 electrodes, 5: 5 frequency bands
        PSD_list: power spectral density, (5, 50, 62, 5)
            5: 5 videos, 50: 50 clips per video, 62: 62 electrodes, 5: 5 frequency bands
    '''
    DE_list = []
    PSD_list = []
    for i in range(0,data.shape[0]):
        DE_list_i = []
        PSD_list_i = []
        for j in range(0, data.shape[1]):
            DE, PSD = DE_PSD(data[i,j,:], time_win)
            DE_list_i.append(DE)
            PSD_list_i.append(PSD)
        DE_list.append(np.stack(DE_list_i, 0))
        PSD_list.append(np.stack(PSD_list_i, 0))
    DE_list = np.stack(DE_list, 0)
    PSD_list = np.stack(PSD_list, 0)
    return DE_list, PSD_list

def save_DE_PSD(data_dir, PSD_save_dir, DE_save_dir, time_win):
    data_list = os.listdir(data_dir)
    for file in tqdm(data_list):
        data_path = os.path.join(data_dir, file)
        PSD_save_path = os.path.join(PSD_save_dir, file.replace('.npy', '_PSD.npy'))
        DE_save_path = os.path.join(DE_save_dir, file.replace('.npy', '_DE.npy'))
        data = np.load(data_path)
        DE_list, PSD_list = DE_PSD_clip(data, time_win)
        np.save(PSD_save_path, DE_list)
        np.save(DE_save_path, PSD_list)



if __name__ == '__main__':
    watching_file = os.getenv('extract_DE_PSD_watching_file')
    watching_file_save_PSD = os.getenv('extract_DE_PSD_watching_file_save_PSD')
    watching_file_save_DE = os.getenv('extract_DE_PSD_watching_file_save_DE')
    imaging_file = os.getenv('extract_DE_PSD_imaging_file')
    imaging_file_save_PSD = os.getenv('extract_DE_PSD_imaging_file_save_PSD')
    imaging_file_save_DE = os.getenv('extract_DE_PSD_imaging_file_save_DE')
    save_DE_PSD(watching_file, watching_file_save_PSD, watching_file_save_DE, time_win=2)
    save_DE_PSD(imaging_file, imaging_file_save_PSD, imaging_file_save_DE, time_win=3)


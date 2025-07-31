'''
this script is used to concatenate PSD and DE data from extract_DE_PSD.py
'''

import numpy as np
import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv('variable.env')

watching_path = os.getenv('concat_watch_image_watching_path')
image_path = os.getenv('concat_watch_image_imaging_path')

def concat_watch_image(path):
    '''
    Args:
        path: PSD path of watching or imaging data, used to load data from extract_DE_PSD.py
    Save: 
        PSD_DE data: concatenated PSD and DE data, (2, 5, 50, 62, 5)
            2: PSD and DE data, 5: 5 videos, 50: 50 clips, 62: 62 electrodes, 5: 5 frequency bands
    '''
    PSD_path = os.path.join(path, 'PSD')
    DE_path = os.path.join(path, 'DE')
    for file in tqdm(os.listdir(PSD_path)):
        PSD_file_path = os.path.join(PSD_path, file)
        DE_file_path = os.path.join(DE_path, file).replace('PSD.npy', 'DE.npy')
        PSD_data = np.load(PSD_file_path)
        DE_data = np.load(DE_file_path)
        data = np.stack((PSD_data, DE_data))
        np.save(os.path.join(path, 'PSD_DE', file).replace('PSD.npy', '_PSD_DE.npy'), data)

if __name__ == '__main__':
    concat_watch_image(watching_path)
    concat_watch_image(image_path)
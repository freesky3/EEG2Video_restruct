import numpy as np
import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv('variable.env')

watching_path = os.getenv('concat_watch_image_watching_path')
image_path = os.getenv('concat_watch_image_imaging_path')

def concat_watch_image(path):
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
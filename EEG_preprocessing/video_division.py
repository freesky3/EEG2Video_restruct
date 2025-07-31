'''
divide the watching and imaging data corresponding to each clip in each video.
input data from slice_video.py
'''

import numpy as np
import os
from dotenv import load_dotenv

load_dotenv("variable.env")

# set the path of the input file and output files
input_file = os.getenv("video_division_input_file")
watching_file = os.getenv("video_division_watching_file")
imaging_file = os.getenv("video_division_imaging_file")

def divide_watch_image(input_file):
    '''used to divide the watching and imaging data corresponding to each clip in each video.
    Args:
        input_file: the input file of slice_video.py, shape is (5, 62, 505*200)
            5: 5 videos, 62: 62 electrodes, 505*200: 505 seconds * 200 Hz
    Returns:
        watching_data: the watching data corresponding to each video, shape is (5, 50, 62, 2*200)
            5: 5 videos, 50: 50 clips per video, 62: 62 electrodes, 400: 2*200: 2 seconds * 200 Hz
        imaging_data: the imaging data corresponding to each video, shape is (5, 50, 62, 3*200)
            5: 5 videos, 50: 50 clips per video, 62: 62 electrodes, 600: 3*200: 3 seconds * 200 Hz
    '''
    watching_start = np.array([i+5 for i in range(0, 500, 10)])
    watching_range = zip(watching_start, watching_start+2)
    imaging_start = watching_start + 4
    imaging_range = zip(imaging_start, imaging_start+3)
    fre = 200
    watching_data = []
    for start, end in watching_range:
        watching_data.append(input_file[:, :, start*fre:end*fre])
    watching_data = np.stack((watching_data), axis=1)
    imaging_data = []
    for start, end in imaging_range:
        imaging_data.append(input_file[:, :, start*fre:end*fre])
    imaging_data = np.stack((imaging_data), axis=1)
    return watching_data, imaging_data

from tqdm import tqdm

# load the input file
file_list = [f for f in os.listdir(input_file) if f.endswith('.npy')]
for f in tqdm(file_list):
    tqdm.write(f"processing file: {f}")
    data = np.load(os.path.join(input_file, f))
    # divide the data corresponding to each video
    watching_data, imaging_data = divide_watch_image(data)
    # save the output files
    np.save(os.path.join(watching_file, f), watching_data)
    np.save(os.path.join(imaging_file, f), imaging_data)
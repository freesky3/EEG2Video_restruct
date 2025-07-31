'''
input the result of slice_video.py, and divide the data corresponding to each video. 
the input shape is (5, 62, 101000)
there are two output files: 
    1. watching data: (5, 50, 62, 400)
    2. imaging data: (5, 50, 62, 600)
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
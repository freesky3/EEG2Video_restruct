'''
This script is used to slice EEG data into small segments between events, and desample the data to 200Hz.
Event's id = 1 represents the start of a video segment, 
event's id = 2 represents the end of a video segment.
This script will slice all the *.cnt files in the EEG_data folder and save the sliced data into EEG_data_slice folder.
The output data's shape is (n_segments, n_channels, n_samples)(5, 66, 101000)
'''

import mne
import numpy as np

def get_raw(fname):
    '''get raw data from cnt file'''
    raw = mne.io.read_raw_cnt(fname, preload=True)
    return raw

def drop_bad(raw, bad_chans):
    raw.drop_channels(bad_chans)
    return raw

from scipy.signal import resample
def resample_data(data, downsample_ratio):
    if downsample_ratio == 2.5:
        resample_len = data.shape[1] * 2 //5
    elif downsample_ratio == 5:
        resample_len = data.shape[1] // 5
    else:
        raise ValueError("downsample_ratio should be 2.5 or 5")
    resample_data = resample(data, resample_len, axis=1)
    return resample_data


def slice_EEG(raw, min_length=505000, downsample_ratio=5):
    events, _ = mne.events_from_annotations(raw)
    # filter events by idp
    events = np.array([i for i in events if i[2] == 1 or i[2] == 2])
    
    start_events = events[0::2, 0]
    end_events = events[1::2, 0]
    time_range = list(zip(start_events, end_events))
    min_length = min_length 

    slice_list = []

    for start, end in time_range:
        data_slice = raw.get_data(start=start, stop=end)
        channel_data = data_slice[:, :min_length]
        channel_data = resample_data(channel_data, downsample_ratio)
        slice_list.append(channel_data)
        slice_array = np.stack(slice_list, axis=0)
    return slice_array

import re
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("variable.env")

if __name__ == '__main__':
    bad_chans = ['M1', 'M2', 'VEO', 'HEO']
    file_path = os.getenv("slice_video_file_path")
    save_path = os.getenv("slice_video_save_path")
    file_list = [f for f in os.listdir(file_path) if f.endswith('.cnt')]
    for f in tqdm(file_list):
        # skip gengtingrui_20250710_session3.cnt because it has some problems
        if f == 'gengtingrui_20250710_session3.cnt':
            continue
        tqdm.write("processing file: {}".format(f))
        fname = os.path.join(file_path, f)
        raw = get_raw(fname)
        raw = drop_bad(raw, bad_chans)

        # when conducting experiments on zhangyiran_20250722_session3.cnt, 
        # we use frequency 500 instead of 1000, so we need to slice the data with a smaller length
        if f == 'zhangyiran_20250722_session3.cnt':
            slice_data = slice_EEG(raw, min_length=252500, downsample_ratio=2.5)
        else:
            slice_data = slice_EEG(raw)
        print(slice_data.shape)
        modified_path = re.sub(r'\.cnt$', '.npy', fname)
        np.save(os.path.join(save_path, os.path.basename(modified_path)), slice_data)
    
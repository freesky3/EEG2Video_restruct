import numpy as np
from einops import rearrange
import os
from einops import repeat
from dotenv import load_dotenv

load_dotenv("variable.env")

def data_label(data_path, label_path):
    data = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        data.append(np.load(file_path))
    data = np.stack(data, axis=0)
    label = np.load(label_path)
    rearrange_data = rearrange(data, 'a b c d e f -> (a b c d) (e f)')
    rearrange_label = repeat(label, 'a b -> (repeat a b)', repeat=110)
    data_type = [
        ('features', rearrange_data.dtype, (rearrange_data.shape[1],)), 
        ('label', rearrange_label.dtype)
    ]

    structured_data = np.empty(27500, dtype=data_type)
    structured_data['features'] = rearrange_data
    structured_data['label'] = rearrange_label
    return structured_data


if __name__ == '__main__':
    watching_path = os.getenv('structured_data_watching_path')
    imaging_path = os.getenv('structured_data_imaging_path')
    label_path = os.getenv('structured_data_label_path')
    save_path = os.getenv('structured_data_save_path')
    watching_structured_data = data_label(watching_path, label_path)
    imaging_structured_data = data_label(imaging_path, label_path)
    np.save(os.path.join(save_path, "watching_structured_data.npy"), watching_structured_data)
    np.save(os.path.join(save_path, "imaging_structured_data.npy"), imaging_structured_data)


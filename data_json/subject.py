'''
This file is used to generate a json file containing the subject information. 
'''

import json
import os
import re

# set your date file path
data_file = # todo r'.\data_json'
# get all file names in the data_file
file_names = os.listdir(data_file)

subject_dict = {}

# the file_name should be in the format of 'Subject_Time_Num.xxx'
for file_name in file_names:
    parts = file_name.split('_')
    # ensure the format is correct
    if len(parts) == 3:
        subject_name = parts[0]
        time_test = parts[1]
        num_test = parts[2].split('.')[0]
        if subject_name not in subject_dict:
            subject_dict[subject_name] = {num_test: time_test}
        else:
            subject_dict[subject_name][num_test] = time_test
    else:
        raise ValueError('Invalid file name: {}'.format(file_name))

# save the subject_dict to a json file
with open(r'.\subject.json', 'w', encoding='utf-8') as json_file:
    json.dump(subject_dict, json_file, ensure_ascii=False, indent=4)

print('Subject data saved to subject.json')

'''
the format of the json file is:
{
    "subject_name": {
        "num_test1": "time_test1"
        "num_test2": "time_test2"
        "num_test3": "time_test3"
    }

    "subject_name2": {
        "num_test1": "time_test1"
        "num_test2": "time_test2"
        "num_test3": "time_test3"
    }

    ...
}
'''
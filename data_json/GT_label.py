'''
generate GT_label.npy
GT_label.npy is a (5, 50) numpy array, 
where 5 is the number of groups, 50 is the number of videos in each group, 
and each element is the label of the corresponding video.
'''

'''
自然景观：
    动物：1
    植物：2
    水：3
    山脉：4
    天气：5
人类行为： 
    笑容：6
    跑步：7
    看书：8
    交谈：9
    吃饭：10
人造物品：
    电子产品：11
    家具：12
    交通工具：13
    衣物：14
    娱乐用品：15
复合场景：
    会议：16
    节日：17
    竞赛：18
    游行：19
    灾难：20
'''

import numpy as np
import json
from hash_label_dict import get_file_hash

import os 
from dotenv import load_dotenv
load_dotenv("variable.env")

hash_label = json.load(open('hash_label.json', 'r'))


label = np.zeros((5, 50))
file_path = os.getenv('GT_label_file_path')

# generate label for each video
for i in range(1, 251):
    num_group = ((i-1) // 50) +1
    num_index = (i-1) % 50
    video_path = file_path + "\\" + "group" + str(num_group) + "\\" + str(i) + ".mp4"
    video_hash = get_file_hash(video_path)
    label[num_group-1, num_index] = hash_label[video_hash]

print(label)
np.save('GT_label.npy', label)
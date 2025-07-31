'''
generate GT_label.npy
GT_label.npy is a (5, 50) numpy array, 
5 videos per experiment, 50 clips per video, each clips is 2 seconds long.
and each element is the label of the corresponding video.
'''

'''
自然景观 (Natural Landscape):
    动物 (Animals): 1
    植物 (Plants): 2
    水 (Water): 3
    山脉 (Mountains): 4
    天气 (Weather): 5
人类行为 (Human Activities): 
    笑容 (Smiling): 6
    跑步 (Running): 7
    看书 (Reading): 8
    交谈 (Conversation): 9
    吃饭 (Eating): 10
人造物品 (Man-made Objects):
    电子产品 (Electronics): 11
    家具 (Furniture): 12
    交通工具 (Vehicles): 13
    衣物 (Clothing): 14
    娱乐用品 (Recreational Items): 15
复合场景 (Complex Scenes):
    会议 (Meeting): 16
    节日 (Festival): 17
    竞赛 (Competition): 18
    游行 (Parade): 19
    灾难 (Disaster): 20
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
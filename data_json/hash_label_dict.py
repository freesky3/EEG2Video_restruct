'''
used to generate the hash_label_dict.json file
each video file's hash value will correspond to a label
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


import hashlib
import json
import os
from dotenv import load_dotenv

load_dotenv("variable.env")

def get_file_hash(file_path):
    """caculate the file's SHA256 hash value"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            sha256.update(chunk)
    return sha256.hexdigest()

video1_hash = get_file_hash(os.getenv("hash_label_dict_video1_hash"))
video2_hash = get_file_hash(os.getenv("hash_label_dict_video2_hash"))

dict1 = {}

file_path1 = os.getenv("hash_label_file_path1")
for file in os.listdir(file_path1):
    if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(1, 11)]:
        dict1[get_file_hash(os.path.join(file_path1, file))] = 1
    if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(11, 21)]:
        dict1[get_file_hash(os.path.join(file_path1, file))] = 2
    if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(21, 31)]:
        dict1[get_file_hash(os.path.join(file_path1, file))] = 3
    if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(31, 41)]:
        dict1[get_file_hash(os.path.join(file_path1, file))] = 4
    if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(41, 51)]:
        dict1[get_file_hash(os.path.join(file_path1, file))] = 5

file_path2 = os.getenv("hash_label_file_path2")
for file in os.listdir(file_path2):
    if "laugh" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path2, file))] = 6
    if "jog" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path2, file))] = 7
    if "read" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path2, file))] = 8
    if "talk" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path2, file))] = 9
    if "eat" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path2, file))] = 10

file_path3 = os.getenv("hash_label_file_path3")
for file in os.listdir(file_path3):
    if "电子产品" in os.path.basename(file):# 人造物品
        dict1[get_file_hash(os.path.join(file_path3, file))] = 11
    if "家具" in os.path.basename(file):# 家具
        dict1[get_file_hash(os.path.join(file_path3, file))] = 12
    if "交通工具" in os.path.basename(file):# 交通工具
        dict1[get_file_hash(os.path.join(file_path3, file))] = 13
    if "衣物" in os.path.basename(file):# 衣物
        dict1[get_file_hash(os.path.join(file_path3, file))] = 14
    if "娱乐用品" in os.path.basename(file):# 娱乐用品
        dict1[get_file_hash(os.path.join(file_path3, file))] = 15

file_path4 = os.getenv("hash_label_file_path4")
for file in os.listdir(file_path4):
    if "会议" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path4, file))] = 16
    if "节日" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path4, file))] = 17
    if "竞赛" in os.path.basename(file) or "contest" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path4, file))] = 18
    if "游行" in os.path.basename(file) or "demonstration" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path4, file))] = 19
    if "灾难" in os.path.basename(file) or "disaster" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path4, file))] = 20

file_path5 = os.getenv("hash_label_file_path5")
for file in os.listdir(file_path5):
    if "新视频" in os.path.basename(file):
        if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(1, 6)]:# 动物
            dict1[get_file_hash(os.path.join(file_path5, file))] = 1
        if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(6, 10)]:# 植物
            dict1[get_file_hash(os.path.join(file_path5, file))] = 2
        if os.path.basename(file).split(".")[0].split("-")[-1] in [str(i) for i in range(10, 16)]:# 天气
            dict1[get_file_hash(os.path.join(file_path5, file))] = 5
    #=================================================================
    if "laugh" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 6
    if "jog" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 7
    if "read" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 8
    if "talk" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 9
    if "eat" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 10
    #=================================================================
    if "人造物品" in os.path.basename(file):
        if os.path.basename(file).split(".")[0].split("-")[-1] in ['7', '8', '9', '10', '13']:# 电子产品
            dict1[get_file_hash(os.path.join(file_path5, file))] = 11
        if os.path.basename(file).split(".")[0].split("-")[-1] in ['11', '12']:# 家具
            dict1[get_file_hash(os.path.join(file_path5, file))] = 12
        if os.path.basename(file).split(".")[0].split("-")[-1] in ['1', '5', '6']:# 交通工具
            dict1[get_file_hash(os.path.join(file_path5, file))] = 13
        if os.path.basename(file).split(".")[0].split("-")[-1] in ['4']:# 衣物
            dict1[get_file_hash(os.path.join(file_path5, file))] = 14
        if os.path.basename(file).split(".")[0].split("-")[-1] in ['2', '3']:# 娱乐用品
            dict1[get_file_hash(os.path.join(file_path5, file))] = 15
    #=================================================================
    if "meeting" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 16
    if "festival" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 17
    if "contest" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 18
    if "disaster" in os.path.basename(file):
        dict1[get_file_hash(os.path.join(file_path5, file))] = 20
        
print(len(dict1))
# save dict1 to the hash_label_dict.json file
with open(r'.\hash_label.json', 'w', encoding='utf-8') as json_file:
    json.dump(dict1, json_file, ensure_ascii=False, indent=4)

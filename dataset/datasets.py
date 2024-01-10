import os
import json
from collections import Counter

import torch
import cv2
from torch.utils.data import Dataset
from torch.nn import functional as F


class FSTDataset(Dataset):

    def __init__(self, image_folder, split, json_file, binary):

        self.image_folder = image_folder
        self.split = split
        self.json_file = json_file
        self.binary = binary

        self.data_list = self._load_json_db(self.json_file)

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_db = json.load(fid)

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        label_id_list = []
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            if self.binary:
                label_id = 0 if value["label_id"] == 0 else 1
            else:
                label_id = value["label_id"]

            label_id_list.append(label_id)

            dict_db += ({'id': key,
                         'label_id': label_id},)
        
        self.id_counts = Counter(label_id_list)
        
        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        item = self.data_list[idx]
        key = item["id"]
        label_id = item["label_id"]
        image_01 = torch.from_numpy(
            cv2.imread(os.path.join(self.image_folder, "Back_BF", "match_Back_BF_{}.bmp".format(key)),
                       cv2.IMREAD_GRAYSCALE)).float()
        image_01 = F.interpolate(image_01[None, None], size=(224, 224), mode="bilinear").squeeze()
        image_02 = torch.from_numpy(
            cv2.imread(os.path.join(self.image_folder, "Back_DF", "match_Back_DF_{}.bmp".format(key)),
                       cv2.IMREAD_GRAYSCALE)).float()
        image_02 = F.interpolate(image_02[None, None], size=(224, 224), mode="bilinear").squeeze()
        image_03 = torch.from_numpy(
            cv2.imread(os.path.join(self.image_folder, "Front_BF", "Front_BF_{}.bmp".format(key)),
                       cv2.IMREAD_GRAYSCALE)).float()
        image_03 = F.interpolate(image_03[None, None], size=(224, 224), mode="bilinear").squeeze()
        image_04 = torch.from_numpy(
            cv2.imread(os.path.join(self.image_folder, "Front_DF", "Front_DF_{}.bmp".format(key)),
                       cv2.IMREAD_GRAYSCALE)).float()
        image_04 = F.interpolate(image_04[None, None], size=(224, 224), mode="bilinear").squeeze()
        torch_image = torch.stack((image_01, image_02, image_03, image_04), axis=0)
        torch_image = (torch_image / 255.0) * 2.0 - 1.0

        return torch_image, label_id
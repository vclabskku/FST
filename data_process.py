import json
import csv
import random
from utils.opt import get_opts

args = get_opts()

label_dict = {"x": 0, "f": 1, "b": 2, "d": 3}
classwise_groups = [list() for _ in range(len(label_dict))]
with open(args.csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        roi_num = row["ROI Number"]
        label = row["Label_SJ"].replace(" matching fail", "").replace("_matching fail", "")
        if label not in label_dict.keys():
            continue
        label_id = label_dict[label]
        classwise_groups[label_id].append(row)

ratio = 2.0 / 3.0
train_set = list()
val_set = list()
for group in classwise_groups:
    train_indices = random.sample(range(len(group)), round(len(group) * ratio))
    val_indices = set(range(len(group))) - set(train_indices)

    for idx in train_indices:
        train_set.append(group[idx])
    for idx in val_indices:
        val_set.append(group[idx])

meta_dict = dict()
for row in train_set:
    roi_num = row["ROI Number"]
    label = row["Label_SJ"].replace(" matching fail", "").replace("_matching fail", "")
    label_id = label_dict[label]
    meta_dict[roi_num] = dict()
    meta_dict[roi_num]["label"] = label
    meta_dict[roi_num]["label_id"] = label_id
    meta_dict[roi_num]["subset"] = "train"

for row in val_set:
    roi_num = row["ROI Number"]
    label = row["Label_SJ"].replace(" matching fail", "").replace("_matching fail", "")
    label_id = label_dict[label]
    meta_dict[roi_num] = dict()
    meta_dict[roi_num]["label"] = label
    meta_dict[roi_num]["label_id"] = label_id
    meta_dict[roi_num]["subset"] = "val"

with open(args.meta_path, "w") as fp:
    json.dump(meta_dict, fp, indent=4, sort_keys=True)
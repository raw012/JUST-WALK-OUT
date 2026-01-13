# This file aims to handle COCO dataset from Kaggle
#The dataset is not downloaded automatically.
#Please download COCO 2017 manually and provide paths via command-line arguments.
import os, json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

COCO_FRUIT_ID_TO_IDX = {52: 0, 53: 1, 55: 2}  # banana, apple, orange

def encode_yolo_target(boxes, labels, img_w, img_h, S=7, C=3):
    target = torch.zeros((S, S, 5 + C))
    cell_w, cell_h = img_w / S, img_h / S

    for box, label in zip(boxes, labels):
        if label not in COCO_FRUIT_ID_TO_IDX:
            continue
        cls = COCO_FRUIT_ID_TO_IDX[label]

        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2

        cell_x = min(int(cx / cell_w), S - 1)
        cell_y = min(int(cy / cell_h), S - 1)

        if target[cell_y, cell_x, 0] == 1:
            continue  # YOLO v1: one object per cell

        target[cell_y, cell_x, 0] = 1
        target[cell_y, cell_x, 1] = (cx - cell_x * cell_w) / cell_w
        target[cell_y, cell_x, 2] = (cy - cell_y * cell_h) / cell_h
        target[cell_y, cell_x, 3] = w / img_w
        target[cell_y, cell_x, 4] = h / img_h
        target[cell_y, cell_x, 5 + cls] = 1

    return target

class COCOFruitDataset(Dataset):
    """
    COCO-style dataset loader for YOLO training.

    This class expects:
    - image_dir: directory containing COCO images (e.g. train2017/)
    - annotation_file: COCO instances_train2017.json

    Note:
    This dataset loader does NOT download data automatically.
    Users are expected to download COCO dataset separately
    (e.g. via Kaggle COCO website).
    """
        
    def __init__(self, image_dir, annotation_file, S=7, C=3):
        self.image_dir = image_dir
        self.S = S
        self.C = C
        self.transform = transforms.ToTensor()

        with open(annotation_file) as f:
            data = json.load(f)

        self.imgs = {img["id"]: img["file_name"] for img in data["images"]}
        self.instances = {}

        for ann in data["annotations"]:
            if ann["category_id"] not in COCO_FRUIT_ID_TO_IDX:
                continue
            self.instances.setdefault(ann["image_id"], []).append(
                (ann["bbox"], ann["category_id"])
            )

        self.image_ids = list(self.instances.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, self.imgs[img_id])
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        boxes, labels = zip(*self.instances[img_id])
        target = encode_yolo_target(boxes, labels, img_w, img_h, self.S, self.C)
        image = self.transform(image)

        return image, target

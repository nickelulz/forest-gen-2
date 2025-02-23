import os
from dataclasses import dataclass

import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Dataset, DataLoader

from util import get_transform
import utils

class ForestDataset(Dataset):
    IMAGE_SIZE = 640

    def __init__(self, root, transforms, limit=None):
        self.root = root
        self.transforms = transforms

        all_in_folder = lambda subdir: list(sorted(os.listdir(os.path.join(root, subdir))))
        self.images = all_in_folder('images') 
        self.labels = all_in_folder('labels')

        if limit is not None:
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]

    @classmethod
    def line_to_bbox(self, line):
        """
        Converts a line of normalized coordinates
        to a bounding box.
        """
        parts = list(map(float, line.strip().split()))
        coords = parts[1:]
        x_coords = coords[0::2]
        y_coords = coords[1::2]

        # Convert polygon to bounding box
        xmin = min(x_coords) * self.IMAGE_SIZE
        ymin = min(y_coords) * self.IMAGE_SIZE
        xmax = max(x_coords) * self.IMAGE_SIZE
        ymax = max(y_coords) * self.IMAGE_SIZE
        
        height = ymax - ymin
        width = xmax - xmin

        if height <= 0 or width <= 0:
            print('Invalid bbox, Skipping: ', line)
            return None, None

        area = height * width

        return [xmin, ymin, xmax, ymax], area

    def __getitem__(self, index):
        image_path = os.path.join(self.root, 'images', self.images[index])
        label_path = os.path.join(self.root, 'labels', self.labels[index])
        image = read_image(image_path)

        with open(label_path, 'r') as f:
            lines = f.readlines()
            bboxes = []
            areas = []

            for line in lines:
                bbox, area = self.line_to_bbox(line)
                if bbox != None:
                    bboxes.append(bbox)
                    areas.append(area)

            num_bboxes = len(bboxes)

            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor([1] * num_bboxes, dtype=torch.int64)
            is_crowd = torch.zeros((num_bboxes), dtype=torch.int64) 
            image_id = torch.tensor([index]) 
            areas = torch.as_tensor(areas, dtype=torch.float32)

            target = {
                "boxes": tv_tensors.BoundingBoxes(bboxes, 
                                                  format="XYXY", 
                                                  canvas_size=F.get_size(image)),
                "labels": labels,
                "image_id": image_id,
                "area": areas,
                "iscrowd": is_crowd
            }

            if self.transforms:
                image, target = self.transforms(image, target)

            return image, target

    def __len__(self):
        return len(self.images)

@dataclass
class DatasetLoaderWrapper:
    train: DataLoader 
    test:  DataLoader
    valid: DataLoader

def load_dataset(dataset_root_path, limit=None) -> DataLoader:
    """
    Loads the dataset (and loader) from a path
    """
    train = DataLoader(
        ForestDataset(os.path.join(dataset_root_path, 'train'), 
                      get_transform(train=True),
                      limit),
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    valid = DataLoader(
        ForestDataset(os.path.join(dataset_root_path, 'valid'), 
                      get_transform(train=False)),
        batch_size=1,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    test = DataLoader(
        ForestDataset(os.path.join(dataset_root_path, 'test'), 
                      get_transform(train=False)),
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    return DatasetLoaderWrapper(train, test, valid)

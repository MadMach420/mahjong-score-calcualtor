import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame
from PIL import Image



class TileDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

        with open(f"{label_path}/_annotations.coco.json", 'r') as f:
            info = json.load(f)
            self.annotations = DataFrame(info['annotations'])
            self.images = DataFrame(info['images'])

    def __getitem__(self, idx):
        img_info = self.images[self.images["id"] == idx].iloc[0]
        image = Image.open(f"{self.image_path}/{img_info['file_name']}").convert('RGB')
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)

        annotations: DataFrame = self.annotations[self.annotations["image_id"] == img_info['id']]
        boxes = np.array([np.array(i) for i in annotations['bbox'].values])
        labels = np.array(annotations['category_id'], dtype=np.int64)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    train_dataset = TileDataset(image_path='./train', label_path='./train')
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    image, target = next(iter(dataloader))
    print((max(image[0][0][0])))
    print(target)

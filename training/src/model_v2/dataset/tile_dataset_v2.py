import os.path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import ToDtype, Resize

from training.src.model_v2 import env
from training.src.model_v2.iou import match_anchor_box


class TileDatasetV2(Dataset):
    def __init__(self, path):
        self.path = path
        self.images: list[str] = []

        files = os.listdir(path)
        for file in files:
            if file.endswith(".jpg"):
                self.images.append(os.path.join(path, file))

    def __getitem__(self, index):
        img, labels, bboxes = self._get_sample(index)
        _, height, width = img.size()
        target = self._convert_target(labels, bboxes, width, height)
        return img, target

    def _get_sample(self, index):
        img_path = self.images[index]
        target_path = img_path[:-4] + ".txt"

        img = read_image(img_path)
        img = Resize((env.WIDTH, env.HEIGHT))(img)
        img = ToDtype(torch.float32)(img)
        img /= 255

        labels = []
        bboxes = []
        with open(target_path, "r") as f:
            for line in f:
                values = line.split()
                labels.append(int(values[0]))
                bboxes.append([float(values[1]), float(values[2]), float(values[3]), float(values[4])])

        return img, labels, bboxes

    def _convert_target(self, labels, bboxes, width, height):
        # Convert target to format expected by loss function
        target = torch.zeros(env.S, env.S, env.NO_OF_ANCHOR_BOX, 5 + env.NO_OF_CLASS)

        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox
            i, j = (int(x * env.S), int(y * env.S))
            x_cell, y_cell = (x * env.S - i, y * env.S - j)
            w_cell, h_cell = (w * env.S, h * env.S)

            anchor_box_index = match_anchor_box(w_cell, h_cell)
            anchor_box = env.ANCHOR_BOXES[anchor_box_index]
            width, height = w_cell / anchor_box[0], h_cell / anchor_box[1]

            target[i, j, anchor_box_index, :5] = torch.tensor([x_cell, y_cell, width, height, 1])
            target[i, j, anchor_box_index, 5 + label] = 1

        return target

    def inverse_target(self, ground_truth):
        # Copied from kaggle, for displaying boxes and labels on image

        cx = cy = torch.tensor([i for i in range(env.S)]).to(env.DEVICE)
        # for getting the center point of pred bb, bx = sig(tx)+cx in paper
        ground_truth = ground_truth.permute(0, 3, 4, 2, 1)
        ground_truth[..., 1:2, :, :] += cx
        ground_truth = ground_truth.permute(0, 1, 2, 4, 3)
        ground_truth[..., 2:3, :, :] += cy
        ground_truth = ground_truth.permute((0, 3, 4, 1, 2))  # back to B,13,13,5,9
        ground_truth[..., 1:3] *= 32  # to pixels
        # Calculating the height and width in pixels
        # anchor_matrix = torch.empty_like(preds)
        ground_truth[..., 3:5] *= torch.tensor(env.ANCHOR_BOXES).to(env.DEVICE)
        # preds+=anchor_matrix
        ground_truth[..., 3:5] = ground_truth[..., 3:5] * 32  # back to pixel values
        bbox = ground_truth[ground_truth[..., 0] == 1][..., 1:5]
        _, labels = torch.max(ground_truth[ground_truth[..., 0] == 1][..., 5:].view(-1, 4), dim=-1)
        return bbox, labels

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = TileDatasetV2(r"./train")
    img, target = dataset[101]
    # bbox, labels = dataset.inverse_target(target)
    print(img, target)

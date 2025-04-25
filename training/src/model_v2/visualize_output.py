from PIL import Image as PILImage
from PIL.Image import Image

import numpy as np
import torch

from training.src.model_v2 import env
from training.src.model_v2.darknet_backbone import YoloV2
from training.src.model_v2.model_progress_utils import non_max_suppression, process_preds

from torchvision.transforms.v2 import Resize, ToTensor
from torchvision.io import read_image
from torchvision.tv_tensors import TVTensor
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def show(imgs):
    total_images = len(imgs)
    num_rows = (total_images + 1) // 2  # Calculate the number of rows
    #     plt.figure(figsize=(15, ))
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, squeeze=False, figsize=(12, 12))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        row_idx = i // 2
        col_idx = i % 2
        axs[row_idx, col_idx].imshow(np.asarray(img))
        axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def visualize_bb(samples):
    images = []
    for sample in samples:
        img = (sample['image'])
        # img = rev_transform(img)
        img = (img * 225).to(torch.uint8)
        bboxes = sample['bbox']
        bboxes = bboxes.numpy()
        labels = sample['labels']

        _, height, width, = img.size()

        corr_bboxes = []  # bboxes in the left cornor, rightcornor
        for i, bbox in enumerate(bboxes):
            x, y = bbox[0], bbox[1]  # center of the bounding box
            box_width, box_height = bbox[2], bbox[3]

            # Top left corner of rectangle
            x1 = max(int(x - box_width / 2), 0)
            y1 = max(int(y - box_height / 2), 0)

            # Bottom right corner of rect.
            x2 = min(int(x + box_width / 2), 416)
            y2 = min(int(y + box_height / 2), 416)

            corr_bboxes += [[x1, y1, x2, y2]]

        corr_bboxes = torch.from_numpy(
            np.array(corr_bboxes))  # converting to tensor as draw_bounding_boxes expects tensors
        img_with_bbox = draw_bounding_boxes(img, corr_bboxes, width=3)
        images.append(img_with_bbox)
    show(images)


def visualize_outputs(indices, model, dataset, thres=0.9):
    images_with_bb = []
    for index in indices:
        image, target = dataset[index]
        image = image.to(env.device)
        model = model.to(env.device)

        model.eval()

        preds = model(image.unsqueeze(0))

        preds = process_preds(preds)

        obj = preds[..., 0] > thres

        bboxes = preds[obj][..., 1:5]
        #         print(bboxes[0])
        scores = torch.flatten(preds[obj][..., 0])
        _, ind = torch.max(preds[obj][..., 5:], dim=-1)
        classes = torch.flatten(ind)
        best_boxes = non_max_suppression(bboxes, scores)

        print(scores.shape, torch.sum(obj))

        filtered_bbox = bboxes[best_boxes]
        filtered_classes = classes[best_boxes]

        print(filtered_bbox.shape, filtered_classes.shape)
        if filtered_classes.size(0) > 0:
            sample = {'image': image.detach().cpu(), 'bbox': filtered_bbox.detach().cpu(),
                      'labels': filtered_classes.detach().cpu()}

            images_with_bb.append(sample)

    visualize_bb(images_with_bb)


def visualize_new_image(model, image, thres=0.9,):
    image = letterbox_image(image, (env.W, env.H))
    image /= 255
    image = torch.tensor(image).to(env.device)
    model = model.to(env.device)

    model.eval()

    with torch.no_grad():
        preds = model(image.unsqueeze(0))
    # print(preds.shape)
    preds = process_preds(preds)

    obj = preds[..., 4] > thres
    bboxes = preds[obj][..., 0:4]
    _, ind = torch.max(preds[obj][..., 5:], dim=-1)
    classes = torch.flatten(ind)
    print(classes)
    sample = {'image': image.detach().cpu(), 'bbox': bboxes.detach().cpu(),
              'labels': classes.detach().cpu()}
    visualize_bb([sample])


def letterbox_image(image: TVTensor, size):
    _, ih, iw = image.size()
    target_w, target_h = size
    scale = min(target_w / iw, target_h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized = Resize((nh, nw))(image)
    canvas = np.full((3, target_h, target_w), 128, dtype=np.float32)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[..., top:top + nh, left:left + nw] = resized
    return canvas


if __name__ == "__main__":
    # hehe = PILImage.open("../temp/1.jpg")
    hehe = read_image("../temp/1.jpg")
    model = YoloV2()
    model.load_state_dict(torch.load("../temp/model_4.4.pt"))
    visualize_new_image(model, hehe, thres=0.6)

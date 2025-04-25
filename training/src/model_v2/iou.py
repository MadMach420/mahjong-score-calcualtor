import torch
import training.src.model_v2.env as env


def convert_to_corners(boxes: torch.Tensor):
    x_center, y_center, width, height = boxes.unbind(1)
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return torch.stack((x_min, y_min, x_max, y_max), dim=1)


def match_anchor_box(bbox_w, bbox_h, to_exclude=[], anchor_boxes=env.ANCHOR_BOXES):
    iou = []
    for i, box in enumerate(anchor_boxes):
        if i in to_exclude:
            iou.append(0)
            continue
        intersection_width = min(box[0], bbox_w)  # Scale up as h, w in range 0-13
        intersection_height = min(box[1], bbox_h)
        I = intersection_width * intersection_height
        IOU = I / (bbox_w * bbox_h + box[0] * box[1] - I)
        iou.append(IOU)

    iou = torch.tensor(iou)
    #     print(iou)
    return torch.argmax(iou, dim=0).item()


def intersection_over_union(bb1, bb2):
    bb1 = bb1.to('cpu')
    bb2 = bb2.to('cpu')
    bboxes = torch.vstack((bb1, bb2))
    # Convert center-width-height format to top-left and bottom-right format
    bboxes = convert_to_corners(bboxes)
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bboxes[0]
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bboxes[1]

    # Ensure validity of bounding boxes
    if bb1_x1 > bb1_x2 or bb1_y1 > bb1_y2 or bb2_x1 > bb2_x2 or bb2_y1 > bb2_y2:
        return 0

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


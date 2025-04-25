import torch
import torchvision
from torcheval.metrics import AUC

from torch import Tensor
from training.src.model_v2 import env
from training.src.model_v2.iou import convert_to_corners, intersection_over_union


# Util functions for model training and evaluation, copied from Kaggle


def check_model_accuracy(preds: Tensor, targets, thres=0.5, dataset=None):
    # map = mean_average_precision(preds.clone(), targets.clone(), dataset=dataset, iou_thres_nms=0.6,)


    sig = torch.nn.Sigmoid()

    # No object score will be the recall value

    # Class Score will be the recall

    obj = targets[..., 4] == 1  # mask
    no_obj = targets[..., 4] == 0

    preds[..., 4] = sig(preds[..., 4])

    class_corr = torch.sum((torch.argmax(preds[obj][..., 5:], dim=-1) == torch.argmax(targets[obj][..., 5:], dim=-1)))

    total_class = torch.sum(obj)

    obj_corr = torch.sum(preds[obj][..., 4] > thres)
    total_obj = torch.sum(obj) + 1e-6  # to avoid divide by zero

    no_obj_corr = torch.sum(preds[no_obj][..., 4] < thres)

    total_no_obj = torch.sum(no_obj)

    # return torch.tensor([total_class, class_corr, total_obj, obj_corr, total_no_obj, no_obj_corr, map.item()])
    return torch.tensor([total_class, class_corr, total_obj, obj_corr, total_no_obj, no_obj_corr])


def cal_epoch_acc(total_class_pred, class_corr, total_obj_prd, obj_corr, total_no_obj, no_obj_corr, map=None):
    print('Class Score (R) ', 100 * class_corr / total_class_pred)
    print('Object Score (R) ', 100 * obj_corr / total_obj_prd)
    print('No object Score (R) ', 100 * no_obj_corr / total_no_obj)
    # print('mAP ', map)


def process_preds(preds, anchor_boxes=env.ANCHOR_BOXES, device=env.DEVICE):
    """Takes predictions in float and returns the pixel values as [obj_score, center_cords(x,y), w, h, class_prob]

    Args:
        preds (_type_): shape[B, S, S, N, C+5]
        anchor_boxes (anchor_boxes, optional): _description_. Defaults to ANCHOR_BOXES.
        thre (float, optional): Thershold value to consider prediction as object. Defaults to 0.5.

    Returns:
        tensor: New preds with [conf_score, bbcordi, classes]
    """

    # Calculating the center coordinates of predicted bounding box.
    sig = torch.nn.Sigmoid()
    preds[..., 4] = sig(preds[..., 4])  # objectness score.
    preds[..., 0:2] = sig(preds[..., 0:2])  # sig(tx) in paper, back to pixels from float

    # for getting the center point of pred bb, bx = sig(tx)+cx in paper

    cx = cy = torch.tensor([i for i in range(env.S)], device=device)
    preds = preds.permute((0, 3, 4, 2, 1))  # permute to obtain the shape (B,5,9, 13,13) so that 13,13 can be updated

    preds[..., 0, :, :] += cx
    preds = preds.permute(0, 1, 2, 4, 3)
    preds[..., 1, :, :] += cy
    preds = preds.permute((0, 3, 4, 1, 2))  # bakck to B,13,13,5,9

    preds[..., 0:2] *= 32  # to pixels

    # Calculating the height and width
    preds[..., 2:4] = torch.exp(preds[..., 2:4])  # pw*e^tw in paper
    # anchor_matrix = torch.empty_like(preds)
    preds[..., 2:4] *= torch.tensor(anchor_boxes, device=device)
    # preds+=anchor_matrix
    preds[..., 2:4] = preds[..., 2:4] * 32  # back to pixel values

    return preds


def non_max_suppression(boxes, scores, io_threshold=0.4):
    """
    Perform non-maximum suppression to eliminate redundant bounding boxes based on their scores.

    Args:
        boxes (Tensor): Tensor of shape (N, 4) containing bounding boxes in the format (x_center, y_center, width, height).
        scores (Tensor): Tensor of shape (N,) containing confidence scores for each bounding box.
        threshold (float): Threshold value for suppressing overlapping boxes.

    Returns:
        Tensor: Indices of the selected bounding boxes after NMS.
    """
    # Convert bounding boxes to [x_min, y_min, x_max, y_max] format
    boxes = convert_to_corners(boxes)
    # print(boxes)

    # Apply torchvision.ops.nms
    keep = torchvision.ops.nms(boxes, scores, io_threshold)

    return keep


# iou_thres_for_corr_predn- Min iou with ground bb to consider it as correct prediction.
def mean_average_precision(predictions, targets, dataset, iou_thres_nms=0.6, iou_thres_for_corr_predn=0.6, C=env.C):
    ep = 1e-6
    # getting back pixel values:
    processed_preds = process_preds(predictions).clone()
    pr_matrix = torch.empty(9, C, 2)  # Precision and recall values at 9 different levels of threh(confidance score)

    for thres in range(1, 10, 1):
        ground_truth = targets.clone().to(env.DEVICE)
        conf_thres = thres / 10
        local_pr_matrix = torch.zeros(C, 3)  # Corr_pred, total_preds, ground_truth for every class
        for i in range(processed_preds.size(0)):  # looping over all preds
            # processing the preds to make it suitable
            preds = processed_preds[i]
            obj = preds[..., 4] > conf_thres

            bboxes = torch.flatten(preds[obj][..., 0:4], end_dim=-2)
            scores = torch.flatten(preds[obj][..., 4])
            _, ind = torch.max(preds[obj][..., 5:], dim=-1)
            classes = torch.flatten(ind)

            best_boxes = non_max_suppression(bboxes, scores, iou_thres_nms)

            filtered_bbox = bboxes[best_boxes]
            filtered_classes = classes[best_boxes]

            #         print(filtered_bbox[filtered_classes==0])
            gt_bboxes, labels = dataset.inverse_target(ground_truth[i].unsqueeze(0))  # inverse_target expects batched
            #         print(gt_bboxes, labels)
            # matche the one bbox among the predicted boxes with the ground thruth box that gives higesht iou.
            tracker = torch.zeros_like(labels)  # to keep track of matched boxes

            for c in range(C):
                total_preds = torch.sum(filtered_classes == c)
                corr_preds = 0
                actual_count = torch.sum(labels == c)
                for box in filtered_bbox[filtered_classes == c]:
                    best_iou = 0
                    for index, value in enumerate(labels):
                        if c == value:

                            iou = intersection_over_union(box, gt_bboxes[index])  # format is cx,cy, w,h

                            if iou > best_iou and tracker[index] == 0:
                                best_iou = iou
                                temp = index
                    #
                    if best_iou > iou_thres_for_corr_predn:
                        tracker[temp] = 1
                        corr_preds += 1

                local_pr_matrix[c] += torch.tensor([corr_preds, total_preds, actual_count])

            precision, recall = local_pr_matrix[:, 0] / (local_pr_matrix[:, 1] + ep), local_pr_matrix[:, 0] / (
                    local_pr_matrix[:, 2] + ep)  # pr at a certain threshold c
            #             print(precision, recall) # should be of shape C

            pr_matrix[thres - 1] = torch.cat((precision.view(-1, 1), recall.view(-1, 1)), dim=1)

            # precision_list = torch.nan_to_num(torch.tensor(precision_list), nan = 0)
    pr_matrix = pr_matrix.permute(1, 0, 2)  # now shape class, all pr values

    # calculate the mean precision
    metric = AUC(n_tasks=C)
    metric.update(pr_matrix[..., 0], pr_matrix[..., 1])
    average_precision = metric.compute()

    return average_precision.mean()

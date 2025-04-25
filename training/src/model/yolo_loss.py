import torch


def yolo_loss(predictions, targets, lambda_coord=5, lambda_noobj=0.5):
    print(f"pred: {predictions.shape}")
    print(f"targets: {targets.shape}")

    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_classes = predictions[..., 5:]
    target_boxes = targets[..., :4]
    target_conf = targets[..., 4]
    target_classes = targets[..., 5:]

    # Lambda coord - impacts how important bounding box errors are
    box_loss = lambda_coord * torch.sum((pred_boxes - target_boxes) ** 2)

    obj_loss = torch.sum((pred_conf - target_conf) ** 2)
    # Lambda noobj - decreases importance of boxes without objects
    noobj_loss = lambda_noobj * torch.sum((pred_conf[target_conf == 0]) ** 2)

    classification_loss = torch.sum((pred_classes - target_classes) ** 2)

    loss = box_loss + obj_loss + noobj_loss + classification_loss
    return loss

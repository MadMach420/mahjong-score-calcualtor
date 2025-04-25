import torch.nn as nn

from training.src.model.yolo_backbone import YoloBackbone
from training.src.model.yolo_head import YoloHead


class Yolo(nn.Module):
    def __init__(self, grid_size=14, num_classes=34, num_anchors=3):
        super(Yolo, self).__init__()
        self.backbone = YoloBackbone()
        self.head = YoloHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions

if __name__ == "__main__":
    model = Yolo()
    print(model)

import torch.nn as nn

class YoloHead(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YoloHead, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(128, num_anchors * 5 + num_classes, kernel_size=1)

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()
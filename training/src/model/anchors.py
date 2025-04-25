import numpy as np


def generate_anchors(scales, ratios):
    """
    Generate anchors for a given scale and ratios;
    Anchor boxes are used by Yolo as bounding boxes
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchors.append((width, height))
    return np.array(anchors)
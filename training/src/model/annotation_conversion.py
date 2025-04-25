def convert_to_yolo_format(width, height, bbox):
    """
    Convert bounding box to YOLO format.
    
    (x min, y min, x max, y max) -> (x center, y center, box width, box height)
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return [x_center, y_center, box_width, box_height]
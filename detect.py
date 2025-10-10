# detect.py
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

def detect_objects(image, model_path, conf_thresh=0.5):
    model = YOLO(model_path)
    results = model(image, verbose=False)
    detections = results[0].boxes
    labels = model.names

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
                   (88,159,106), (96,202,231), (159,124,168), (169,162,241),
                   (98,118,150), (172,176,184)]

    for i in range(len(detections)):
        conf = detections[i].conf.item()
        if conf < conf_thresh:
            continue
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        color = bbox_colors[classidx % 10]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        label = f'{classname}: {int(conf*100)}%'
        cv2.putText(image, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


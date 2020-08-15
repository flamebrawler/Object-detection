import tensorflow as tf
import numpy as np
import time

def reduce(detections, accepted_classes, threshold=.3):
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = detections['detection_scores'][0].numpy()
    reduced_list = []
    for box, cls, scr in zip(boxes, classes, scores):
        if cls in accepted_classes and scr > threshold:
            reduced_list.append(((box[0]+box[2])/2, (box[1]+box[3])/2, cls, scr, time.time()*1000.0))
    return reduced_list




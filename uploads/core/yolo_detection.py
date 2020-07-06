#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################

import sys
import os
import cv2
import argparse
import numpy as np
from pprint import pprint


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS, classes):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detection(img, conf="./yolov3-gurmina.cfg", weights="./yolov3-gurmina_best.weights", names="./gurmina.names", detect=0.3, nms=0.1, con=0.2):
    resolution=800

    image = cv2.imread(img)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None
    print("Detection: "+str(detect))
    print("NMS: "+str(nms))
    print("Conf: "+str(con))

    with open(names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))



    res=[416,448,480,512,544,576,608,640,672,704,736,768,800,832,864,896,928,960,992,1024,1056,1088,1120,1152,1184]
    ii=0
    class_ids = []
    confidences = []
    boxes = []
    avg = 0
    while ii < 15:
        ii=ii+1
        resolution=resolution+128
        print(str(resolution)+":"+str(ii))
#        net = cv2.dnn.readNet(weights, conf)
        net = cv2.dnn.readNet(weights, conf)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        blob = cv2.dnn.blobFromImage(image, scale, (int(resolution),int(resolution)), (0,0,0), True, crop=False)

        
        net.setInput(blob)

        outs = net.forward(get_output_layers(net))


        conf_threshold = con
        nms_threshold = nms
        all = []
        a=0

        for out in outs:
            
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > detect:
                    a=a+1
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    if h / w >= 3:
                        h=w*2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    # img=image
                    # draw_prediction(img, class_id, float(confidence), round(x), round(y), round(x+w), round(y+h))
                    # cv2.imshow("object detection", img)
                    print(str(classes[class_id])+ ","+ str(confidence)+", x: "+str(x)+", y: "+str(y))
         if ii > 1:
             
         else:
         

        print(a)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print(len(indices))
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        xc = ( x + w / 2 ) / Width
        yc = ( y + h / 2 ) / Height
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), COLORS, classes)
        print(str(classes[class_ids[i]])+ ","+ str(confidences[i])+", x: "+str(xc)+", yc: "+str(yc))

    cv2.imwrite(img, image)
    return 0

def average(val):
    avg = 0
    for a in len(val):
        avg = avg + a
    avg = avg / len(a)
    return avg

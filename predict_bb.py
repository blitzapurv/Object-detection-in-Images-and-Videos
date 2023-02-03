#! /usr/bin/env python

import os
import time
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np


def make_prediction(input_path):
    DIR_PATH = '/app/' #docker
    #DIR_PATH = './'     #local testing
    #input_path   = video_path
    output_path  = DIR_PATH + 'output/'

    makedirs(output_path)
    start = time.time()
    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.6, 0.3
    anchors = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    
    ###############################
    #   Load the model
    ###############################
    #os.environ['CUDA_VISIBLE_DEVICES'] = 0
    infer_model = load_model("./weights/model.h5")

    ###############################
    #   Predict bounding boxes 
    ###############################
     
    if input_path[-4:] == '.mp4': # do detection on a video 
        #video_out = output_path + input_path.split('/')[-1]
        video_out = output_path + 'output_video.mp4'
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'avc1'), 
                               50.0, 
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, anchors, obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], labels, obj_thresh)   

                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])  

                        # write result to the output video
                        video_writer.write(images[i]) 
                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else: # do detection on an image or a set of images
        #image_out = output_path + input_path.split('/')[-1]
        image_out = output_path + 'output_image.jpg'
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        #image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, labels, obj_thresh) 
     
            # write the image with bounding boxes to file
            cv2.imwrite(image_out, np.uint8(image))         

    end = time.time()
    elap = (end - start)
    return elap


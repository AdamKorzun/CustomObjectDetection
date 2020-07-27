import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import time

from mss import mss

import threading

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



PATH_TO_LABELS = 'labels.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
predict_fn = tf.saved_model.load("model\\saved_model")

def run_inference_for_single_image(model, image):
    

    input_tensor = tf.convert_to_tensor(image)

    input_tensor = input_tensor[tf.newaxis,...]
    ct = time.time()
    output_dict = model(input_tensor)
    print(time.time() - ct)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict
def capture_screen(scr):
    image = scr.grab(mon)
    image_np = np.array(image)
    res = np.shape(image_np)
    image_np = image_np[:,:,:3]
    return image_np
def show_inference(model, np_array):
    image_np = np_array
    output_dict = run_inference_for_single_image(model, image_np)

    vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=8)

    return image_np

if __name__ == '__main__':
    mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    scr = mss()
    frames = 0
    starting_time = time.time()
    while 1:
        image_np = capture_screen(scr)
        #image_np = np.reshape(image_np, (1,res[0],res[1],3))
        new_array =  show_inference(predict_fn,image_np) #!FIX



        res = np.shape(new_array)
        image_np = cv2.resize(new_array, dsize=(int(res[1] / 2),int(res[0] / 2)),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('View', image_np)
        if (time.time() - starting_time > 1):
            #print(frames)
            frames = 0
            starting_time = time.time()
        frames+=1
        if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break




#!/usr/bin/env python
__author__ = "James Burnett"
__copyright__ = "Copyright (C) James Burnett - https://burnett.tech"
__license__ = "GNU AGPLv3"
__maintainer__ = "James Burnett"
__email__ = "james@burnett.tech"
__status__ = "Development"
import imutils
import cv2
import dlib
import datetime
import time
import sys
#import round
from PIL import Image, ImageDraw
from threading import Thread


video_source = sys.argv[1]
svm_file = sys.argv[2]

video = cv2.VideoCapture(video_source)

time.sleep(3)

win = dlib.image_window()

win.clear_overlay()

max_frames = 1

counter = 0


while(1> 0):

    if counter < max_frames:
        counter = counter + 1
        ret, frame = video.read()
        continue
    else:
        ret, frame = video.read()
        counter = 0


    
    
    rgb = frame[:, :, ::-1]
    #rgb = frame

    font = cv2.FONT_HERSHEY_PLAIN
    detector = dlib.simple_object_detector(svm_file)

    dets = detector(rgb)
    if len(dets) > 0:
        win.clear_overlay()
        win.set_image(rgb)
        for d in dets:
            win.add_overlay(d)
    
    
    

    
    rgb2 = frame[:, :, ::-1]
    win.set_image(rgb2)

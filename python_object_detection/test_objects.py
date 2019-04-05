#!/usr/bin/env python
__author__ = "James Burnett"
__copyright__ = "Copyright (C) James Burnett - https://burnett.tech"
__license__ = "GNU AGPLv3"
__maintainer__ = "James Burnett"
__email__ = "james@burnett.tech"
__status__ = "Development"
import os
import sys
import dlib
import inspect
import time
import cv2

image_folder = None

svm_file = None

try:
    image_folder = sys.argv[1]
    svm_file = sys.argv[2]

except IndexError:
    print("usage: ./test_objects <img_folder> <svm detector file>")
    sys.exit();




detector = dlib.simple_object_detector(svm_file)

win_det = dlib.image_window()

win_det.set_image(detector)

print("Showing detections on the images in the faces folder...")
#win = dlib.image_window()
#win_det.set_image(detector)
#input("Press Enter to continue...")

max_detections = 0

for filename in os.listdir(image_folder):
    if max_detections > 10:
        print(max_detections)

    if ".jpg" in filename or ".png" in filename:
        img_file = image_folder + "/" + filename
        #print("Processing file: {}".format(filename))
        img = dlib.load_rgb_image(img_file)
        dets = detector(img)
        if len(dets) > 0:
            win_det.clear_overlay()
            win_det.set_image(img)
            for d in dets:
                win_det.add_overlay(d)

            max_detections = max_detections + 1
            #print("Number of faces detected: {}".format(len(dets)))
print(max_detections)
    #for d in dets[0]:
    #    print("Gun Detected")
    #win_det.clear_overlay()
    #win_det.set_image(img)
    #for d in dets:
        #print("Score: %f" % s)
        #win_det.add_overlay(d)
        #input("Press Enter to continue...")
        
    #if len(dets[0]) > 0:
    #    print("Number of objects detected: {}".format(len(dets[0])))
        #for property, value in vars(dets).iteritems():
        #    print ("%s %s" % ( str(property), str(value)))
        
    #    for d in enumerate(dets[0]):
    #        print("%s" % str(d[1]))
    #        win_det.add_overlay(d[1])
    #    win_det.set_image(img)
    #    input("Press Enter to continue...")

    


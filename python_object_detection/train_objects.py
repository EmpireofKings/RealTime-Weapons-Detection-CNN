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
import cv2

image_folder = None

training_xml = None

svm_file = None

try:
    training_xml = sys.argv[1]
    svm_file = sys.argv[2]
    image_folder = sys.argv[3]

except IndexError:
    print("usage: ./train_objects <training.xml file made with imglab> <svm file to save to (.svm!)")
    sys.exit();


options = dlib.simple_object_detector_training_options()

options.add_left_right_image_flips = True

#options.C = 6 
options.C = 10 

options.epsilon = 0.10

options.num_threads = 10

options.be_verbose = True 

#40 * 40 = 1600
options.detection_window_size = 50*50 

training_xml_path = os.path.join(training_xml)

dlib.train_simple_object_detector(training_xml_path, svm_file, options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(dlib.test_simple_object_detector(training_xml_path, svm_file)))

detector = dlib.simple_object_detector(svm_file)
max_detections = 0
images_tested = 0
win_det = dlib.image_window()
for filename in os.listdir(image_folder):
    if max_detections > 10:
        print(max_detections)

    if ".jpg" in filename or ".png" in filename:
        img_file = image_folder + "/" + filename
        img = dlib.load_rgb_image(img_file)
        dets = detector(img)
        if len(dets) > 0:
            win_det.clear_overlay()
            win_det.set_image(img)
            for d in dets:
                win_det.add_overlay(d)
            max_detections = max_detections + 1
            input("Press Enter to continue...")
        images_tested = images_tested + 1

print("Images Test: %d  Matches: %d" % ( images_tested, max_detections))

import os
import csv
import shutil
import numpy as np
import cv2

eye_corner_left = 39
eye_corner_right = 42
nose_side_left = 31
nose_side_right = 35

def findElementsPositions(csv_header):
    eye_left = []
    eye_left.append(csv_header.index(" X_"+str(eye_corner_left))) # X_
    eye_left.append(csv_header.index(" Y_"+str(eye_corner_left))) # Y_
    eye_left.append(csv_header.index(" Z_"+str(eye_corner_left))) # Z_

    eye_right = []
    eye_right.append(csv_header.index(" X_"+str(eye_corner_right))) # X_
    eye_right.append(csv_header.index(" Y_"+str(eye_corner_right))) # Y_
    eye_right.append(csv_header.index(" Z_"+str(eye_corner_right))) # Z_

    nose_left = []
    nose_left.append(csv_header.index(" X_"+str(nose_side_left))) # X_
    nose_left.append(csv_header.index(" Y_"+str(nose_side_left))) # Y_
    nose_left.append(csv_header.index(" Z_"+str(nose_side_left))) # Z_

    nose_right = []
    nose_right.append(csv_header.index(" X_"+str(nose_side_right))) # X_
    nose_right.append(csv_header.index(" Y_"+str(nose_side_right))) # Y_
    nose_right.append(csv_header.index(" Z_"+str(nose_side_right))) # Z_

    return eye_left, eye_right, nose_left, nose_right

def calculate3DDistance(p1, p2):
    sqrt_dst = np.sum(p1**2 + p2**2, axis=0)
    dst = np.sqrt(sqrt_dst)
    return dst

def loadCSVData(video_path, video_name):
    f = open(video_path+video_name+".csv", "rt")
    reader = csv.reader(f, delimiter=',')
    confidence = []
    success = []
    eye_separation = []
    nose_separation = []

    csv_header = next(reader)
    eye_left, eye_right, nose_left, nose_right = findElementsPositions(csv_header)

    for row in reader:
        confidence.append(float(row[3]))
        success.append(int(row[4]))
        # Store 3D positions of relevant points
        left_eye_pos = [float(row[eye_left[0]]), float(row[eye_left[1]]), float(row[eye_left[2]])]
        right_eye_pos = [float(row[eye_right[0]]), float(row[eye_right[1]]), float(row[eye_right[2]])]

        nose_left_pos = [float(row[nose_left[0]]), float(row[nose_left[1]]), float(row[nose_left[2]])]
        nose_right_pos = [float(row[nose_right[0]]), float(row[nose_right[1]]), float(row[nose_right[2]])]

        # Calculate and save eyes and nose distances (in mm)
        eye_separation.append(calculate3DDistance(np.array(left_eye_pos),np.array(right_eye_pos)))
        nose_separation.append(calculate3DDistance(np.array(nose_left_pos), np.array(nose_right_pos)))

    return np.array(confidence), np.array(success), np.array(eye_separation), np.array(nose_separation)

def checkConfidenceLevels(num_frames, confidence, confid_thres, frames_thres):
    discardVideo = False
    bad_frames = 0
    for element in confidence:
        if element < confid_thres:
            bad_frames += 1
    if float(bad_frames)/float(num_frames) >= frames_thres:
        discardVideo = True
    return discardVideo
    
def checkSuccessLevels(num_frames, success, threshold):
    discardVideo = False
    bad_frames = 0
    for element in success:
        if element != 1:
            bad_frames += 1
    if float(bad_frames)/float(num_frames) >= threshold:
        discardVideo = True
    return discardVideo

def checkShapeDeformation(eyes_sep, nose_sep, distance_thres):
    discardVideo = False
    # If standard deviation of the per-frame distances surpasses 1cm, discard the video
    if np.std(eyes_sep) >= distance_thres or np.std(nose_sep) >= distance_thres:
        discardVideo = True
    return discardVideo

def overUnderExposureFrame(frame, brightness_thres, darkness_thres):
    result = 0 # 0 = image is correctly exposed, 1 = overexposure, 2 = underexposure
    # Load frame image
    img = cv2.imread(frame)
    # Convert to grayscale
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the mask of pixels that are close to the dark and bright ranges
    dark_part = cv2.inRange(frame_gray, 0, 30)
    bright_part = cv2.inRange(frame_gray, 220, 255)
    # Sum the number of pixels
    total_pixels = np.size(frame_gray)
    dark_pixels = np.sum(dark_part > 0)
    bright_pixels = np.sum(bright_part > 0)
    # See if the proportions of dark/bright pixels exceed a certain threshold
    if float(dark_pixels)/total_pixels > darkness_thres:
        result = 2
    if float(bright_pixels)/total_pixels > brightness_thres:
        result = 1
    return result

def overUnderExposure(frames_path, bright_thres, dark_thres, total_bright_th, total_dark_th):
    # Access the folder containing the extracted face bounding box images
    discardVideo = False
    frame_counter = 0
    range_counter = 0
    exp_classif = [0, 0, 0] # Counters for correctly exposed, overexposed and underexposed frames, respectively

    # Get the list of names of the frames image files
    frames_list = os.listdir(frames_path)
    frames_list.sort()
    for frame in frames_list:
        # Add a unit to the corresponding counter, depending on the exposure levels of the frame
        exp_classif[overUnderExposureFrame(frames_path+frame, bright_thres, dark_thres)] += 1

    total_frames = exp_classif[0] + exp_classif[1] + exp_classif[2]
    if total_frames != 0:
        if float(exp_classif[1])/total_frames > total_bright_th or float(exp_classif[2])/total_frames > total_dark_th: # If the video is either overexposed or underexposed, discard it
            discardVideo = True
    else:
        print 'Frames folder is empty!'

    return discardVideo


def verifyQuality(video_path, video_name):
    # Main function where to include calls to rest of image quality verification functions
    discardVideo = False
    frames_path = video_path + video_name + '_aligned/'
    # Load all relevant data from the CSV file
    confidence, success, eyes_sep, nose_sep = loadCSVData(video_path, video_name)

    confidence_thres = 0.85
    conf_frames_th = 0.1
    success_thres = 0.9
    shape_dst_thres = 100.0
    bright_thres = 0.4
    dark_thres = 0.4
    frames_bright_th = 0.2
    frames_dark_th = 0.2

    '''print 'CONFIDENCE LEVELS:'
    print confidence
    print 'NOSE SEPARATION PER FRAME:'
    print nose_sep'''
    num_frames = len(confidence)

    # Start verifying every criterion; if one is not met, directly discard the video
    discardVideo = checkConfidenceLevels(num_frames, confidence, confidence_thres, conf_frames_th)
    print 'Discard video after confidence check: '+str(discardVideo)
    if not discardVideo:
        # Check success levels
        discardVideo = checkSuccessLevels(num_frames, success, success_thres)
        print 'Discard video after success check: '+str(discardVideo)
        if not discardVideo:
            # Check distances between eyes and nostrils
            discardVideo = checkShapeDeformation(eyes_sep, nose_sep, shape_dst_thres)
            print 'Discard video after shape def. check: '+str(discardVideo)
            if not discardVideo:
                discardVideo = overUnderExposure(frames_path, bright_thres, dark_thres, frames_bright_th, frames_dark_th)
                print 'Discard video after img exposure check: '+str(discardVideo)

    return discardVideo


# TEST 1
discardVideo = verifyQuality('data/videos/','testvideo1')
print discardVideo

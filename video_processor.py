import os
import csv
import shutil
import numpy as np
import cv2
import glob

import random
from scipy import ndarray
import skimage as sk

def readAnnotations(annotFile):
    '''
    Read a segmentation ground truth file, based on the format followed
    on the common Excel annotation sheets.
    ARGUMENTS:
     - annotFile:       path of the CSV file
    RETURNS:
     - segStart:     a numpy array of the starting frames for all the expression segments
     - segEnd:       a numpy array of the ending frames for all the expression segments
     - segLabel:     a list of the corresponding class labels
    '''
    f = open(annotFile, "rt")
    reader = csv.reader(f, delimiter=',')
    segStart = []
    segEnd = []
    segLabel = []
    for row in reader:
        if len(row) == 3: # Verify that the row has no missing data
            segStart.append(int(row[0]))
            segEnd.append(int(row[1]))
            segLabel.append(row[2])
    return np.array(segStart), np.array(segEnd), segLabel

def moveFramesNoOverwriting(file_name, orig_dir, dst_dir):
    '''
    Move a frame to its corresponding class subfolder making sure it doesn't overwrite
    other frames with identical names.
    '''
    head, tail = os.path.splitext(file_name)
    # Rename the frame file if necessary
    count = 0
    dst_file = dst_dir+file_name
    while os.path.exists(dst_file):
        count += 1
        dst_file = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))
    shutil.move(orig_dir+file_name,dst_file)


def orderExtractedFrames(video_name,starting_frames,ending_frames,labels):
    '''
    Move the extracted face bounding box images to the corresponding class folders
    depending on the annotated facial expression class.
    '''
    # Access the folder containing the extracted face bounding box images
    frames_folder = video_name+'_aligned'
    frame_counter = 0
    range_counter = 0
    frames_path = "data/videos/"+frames_folder
    # Get the list of names of the frames image files
    frames_list = os.listdir(frames_path)
    frames_list.sort()

    for frame in frames_list:
        '''print "Frame name: "+str(frame)
        print "Frame no.: "+str(frame_counter)
        print "Range: "+str(range_counter)
        print "Current start: "+str(starting_frames[range_counter])
        print "Current end: "+str(ending_frames[range_counter])'''
        # If the frame is not within the current range of annotated frame labels:
        if int(frame_counter) > int(ending_frames[range_counter]):
            # move to the next annotated range
            range_counter += 1
        # Move the frame to its corresponding data class subfolder depending on its label
        #shutil.move(frames_path +"/"+ frame, "data/classes/" + labels[range_counter] + "/" + frame)
        moveFramesNoOverwriting(frame, frames_path +"/", "data/classes/"+labels[range_counter]+"/")
        frame_counter += 1

def preprocessExtractedFrames(frames_path, classes):
    for fields in classes:   
        index = classes.index(fields)
        # The dataset images are stored in subfolders named with the corresponding class label
        # therefore the subfolder name determines the class of each of the loaded images
        path = os.path.join(frames_path, fields, '*.bmp')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(image)
            cv2.imwrite(fl, image)

def dataAugmentNoise(frames_path, classes): # NOT WORKING!
    # Create a new frame with Gaussian noise
    '''meansBGR = [0,0,0]
    sigmasBGR = [10,10,10]
    meansBGR = np.array(meansBGR)
    sigmasBGR = np.array(sigmasBGR)'''
    for fields in classes:   
        index = classes.index(fields)
        path = os.path.join(frames_path, fields, '*.bmp')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            #cv2.randn(noise,meansBGR, sigmasBGR)
            #image += noise
            image = sk.util.random_noise(image)
            frame = os.path.splitext(fl)[0]
            cv2.imwrite(frame+'_noise.bmp', image)

def dataAugmentHFlip(frames_path, classes):
    # Create a new frame horizontally flipped
    for fields in classes:   
        index = classes.index(fields)
        path = os.path.join(frames_path, fields, '*.bmp')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            cv2.flip(src=image,dst=image,flipCode=+1)
            frame = os.path.splitext(fl)[0]
            cv2.imwrite(frame+'_flipped.bmp', image)

# VIDEO PROCESSOR

# Get the list of names of the video files
videos_path = "data/videos/"
videosList = os.listdir(videos_path) # Lists all files (and directories) in the folder
#print videosList
classes = ['concerned','enthusiastic','happy','sad','serious']

for video in videosList:
    if os.path.isfile(os.path.join(videos_path, video)): # Considers files only
        # Process the video through OpenFace
        os.system('./faceDetectorExtraction.sh '+video+' '+videos_path)
        # Read annotations
        video_name = os.path.splitext(video)[0]
        annot_name = 'data/videos/annotations/'+video_name+'_annot.csv'
        starting_frames, ending_frames, labels = readAnnotations(annot_name)
        # Move frames to corresponding class folders
        orderExtractedFrames(video_name,starting_frames,ending_frames,labels)

#dataAugmentNoise('data/classes',classes) # NOT WORKING! TOO NOISY
dataAugmentHFlip('data/classes',classes)
preprocessExtractedFrames('data/classes',classes)

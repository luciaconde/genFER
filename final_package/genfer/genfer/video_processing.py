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

def readAnnotationsMMI(annot_file):
    '''
    Read a segmentation ground truth file, based on the format followed
    on the common Excel annotation sheets.
    ARGUMENTS:
     - annotFile:       path of the CSV file
    RETURNS:
     - video_names:     a numpy array with all the names of the videos
     - labels:       a numpy array of the corresponding labels for each of the videos
    '''
    f = open(annot_file, "rt")
    reader = csv.reader(f, delimiter=',')
    video_names = []
    labels = []
    for row in reader:
        if len(row) == 2: # Verify that the row has no missing data
            video_names.append(row[0])
            labels.append(row[1])
    return video_names, labels

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

def orderExtractedFramesMMI(video_name,data_path,label,step):
    '''
    Move specific extracted face bounding box images (beginning and end: neutral, middle: label)
    to the corresponding class folders depending on the annotated facial expression class.
    '''
    video_path = data_path+'videos/'
    frames_path = video_path+video_name+'_aligned/'
    frames_list = os.listdir(frames_path)
    frames_list.sort()
    mid_frame = int(len(frames_list)/2)-step

    try:
        for frame in range(0,step): # Store the neutral frames
            # Rename frames files
            new_name_init = video_name+"-"+frames_list[frame]
            os.rename(frames_path+frames_list[frame], frames_path+new_name_init)
            moveFramesNoOverwriting(new_name_init, frames_path, data_path+"classes/neutral/")

            new_name_end = video_name+"-"+frames_list[len(frames_list)-1-frame]
            os.rename(frames_path+frames_list[len(frames_list)-1-frame], frames_path+new_name_end)
            moveFramesNoOverwriting(new_name_end, frames_path, data_path+"classes/neutral/")

        for frame in range(0,2*step):
            new_name = video_name+"-"+frames_list[mid_frame+frame]
            try:
                os.rename(frames_path+frames_list[mid_frame+frame], frames_path+new_name)
            except OSError:
                print 'Frame not found (possibly tagged as neutral)'
            moveFramesNoOverwriting(new_name, frames_path, data_path+"classes/"+label+"/")
    except IOError:
        print 'Frame file could not be found!'   

def orderExtractedFramesStep(video_name,data_path,starting_frames,ending_frames,labels,step):
    '''
    Move specific extracted face bounding box images (by step) to the corresponding class folders
    depending on the annotated facial expression class.
    '''
    # Access the folder containing the extracted face bounding box images
    video_path = data_path+'videos/'
    frames_folder = video_name+'_aligned'
    range_counter = 0
    frames_path = video_path+frames_folder+"/"
    # Get the list of names of the frames image files
    frames_list = os.listdir(frames_path)
    frames_list.sort()

    for frame in range(0,len(frames_list),step):
        # If the frame is not within the current range of annotated frame labels:
        if frame > int(ending_frames[range_counter]):
            # move to the next annotated range
            range_counter += 1
        # Rename the frames to keep the video name
        new_name = video_name+"-"+frames_list[frame]
        os.rename(frames_path+frames_list[frame], frames_path+new_name)
        # Move the frame to its corresponding data class subfolder depending on its label
        #shutil.move(frames_path +"/"+ frame, "data/classes/" + labels[range_counter] + "/" + frame)
        moveFramesNoOverwriting(new_name, frames_path, data_path+'classes/'+labels[range_counter]+'/')


def orderExtractedFramesAll(video_name,video_path,starting_frames,ending_frames,labels):
    '''
    Move the extracted face bounding box images to the corresponding class folders
    depending on the annotated facial expression class.
    '''
    # Access the folder containing the extracted face bounding box images
    frames_folder = video_name+'_aligned'
    frame_counter = 0
    range_counter = 0
    frames_path = video_path+frames_folder+"/"
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
        # Rename the frames to keep the video name
        new_name = video_name+"-"+frame
        os.rename(frames_path+frame, frames_path+new_name)
        # Move the frame to its corresponding data class subfolder depending on its label
        #shutil.move(frames_path +"/"+ frame, "data/classes/" + labels[range_counter] + "/" + frame)
        moveFramesNoOverwriting(new_name, frames_path, "data/classes/"+labels[range_counter]+"/")
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
def videoProcessor(data_path, classes):
    # Get the list of names of the video files
    videos_path = data_path+'videos/'
    videosList = os.listdir(videos_path) # Lists all files (and directories) in the folder
    frames_step = 7

    for video in videosList:
        if os.path.isfile(os.path.join(videos_path, video)): # Considers files only
            # Process the video through OpenFace
            os.system('./faceDetectorExtraction.sh '+video+' '+videos_path)
            # Read annotations
            video_name = os.path.splitext(video)[0]
            annot_name = data_path+'annotations/'+video_name+'_annot.csv'
            starting_frames, ending_frames, labels = readAnnotations(annot_name)
            # Move frames to corresponding class folders
            #orderExtractedFramesAll(video_name,videos_path,starting_frames,ending_frames,labels)
            orderExtractedFramesStep(video_name,data_path,starting_frames,ending_frames,labels,frames_step)

    dataAugmentHFlip(data_path+'classes',classes)
    preprocessExtractedFrames(data_path+'classes',classes)

def getVideoLabel(video, video_names, labels):
    try:
        pos = video_names.index(video)
    except ValueError:
        pass
    return labels[pos]

def videoProcessorMMI(data_path, classes):
    # Get the list of names of the video files
    videosList = os.listdir(videos_path) # Lists all files (and directories) in the folder
    videos_path = data_path+'videos/'
    video_names, labels = readAnnotationsMMI(data_path+'annotations/mmi_annot.csv')
    frames_step = 8

    for video in videosList:
        if os.path.isfile(os.path.join(videos_path, video)): # Considers files only
            # Process the video through OpenFace
            os.system('./faceDetectorExtraction.sh '+video+' '+videos_path)
            video_name = os.path.splitext(video)[0]
            label = getVideoLabel(video,video_names,labels)
            # Move frames to corresponding class folders
            orderExtractedFramesMMI(video_name,data_path,label,frames_step)

    dataAugmentHFlip(data_path+'classes',classes)
    preprocessExtractedFrames(data_path+'classes',classes)

def deleteProcessData(videos_path):
    files_list = os.listdir(videos_path) # Lists all files (and directories) in the folder

    for loaded_file in files_list:
        if os.path.isfile(os.path.join(videos_path, loaded_file)):
            if os.path.splitext(loaded_file)[-1].lower()!='.mp4': # Delete all files whose extension is not .mp4
                os.remove(videos_path+loaded_file)
        else: # Delete all subdirectories
            shutil.rmtree(videos_path+loaded_file)

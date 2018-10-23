import numpy as np
import os,glob

import video_predicting as vp
import video_processing as vproc
import quality_verification as qver

# Pre-trained models identifiers
MMI = 0
MMITT = 1
models = ['mmi/','mmitt/']

# Annotation types identifiers
ANNOT_MMI = 0
ANNOT_TT = 1

''' predictVideoDefault: evaluate a single specific video
using one of the default pre-trained models included in genFER'''
def predictVideoDefault(video_file, video_path, nmodel, save_path):
    video_title = os.path.splitext(video_file)[0]
    classes = ['enthusiastic','neutral','concerned']
    path_model = 'defaultmodels/'
    img_height = 90
    img_width = 90

    vp.extractFaceImages(video_file, video_path)

    face_images = vp.loadFaceImages(video_file,video_path)

    labels, confidence = vp.predictVideo(face_images,nmodel,img_height,img_width, classes, path_model, models[nmodel]) # Predict the facial expression label for each of the face images (that is, for each video frame)
#print 'Predicted labels: '+str(labels)
    clean_labels = vp.cleanLabels(labels)
#print 'Cleaned labels: '+str(clean_labels)

    perframe_labels_areStored = vp.savePredictionsPerFrame(clean_labels, save_path + video_title + '_predictions_perframe.csv')
    print perframe_labels_areStored

    perexp_labels_areStored = vp.savePredictionsPerExpression(clean_labels, save_path + video_title + 'predictions_perexp.csv')
    print perexp_labels_areStored

    # Play the tested video with the labels written in text format over the image
    vp.playLabeledVideo(video_path+video_file, clean_labels)

    # Clean all the files created by OpenFace
    vproc.deleteProcessData(video_path)

''' predictVideoDefault: evaluate a set of videos contained in the videos_path directory
using one of the default pre-trained models included in genFER'''
def predictVideoSetDefault(videos_path, nmodel, save_path):
    classes = ['enthusiastic','neutral','concerned']
    img_height = 90
    img_width = 90
    videosList = os.listdir(videos_path) # Lists all files (and directories) in the folder

    for video in videosList:
        if os.path.isfile(os.path.join(videos_path, video)):
            predictVideoDefault(video, videos_path, nmodel, save_path)
    
    # # Clean all the remaining files created by OpenFace
    vproc.deleteProcessData(videos_path)

# CONFIDENCE LEVEL CALCULATION OF VIDEO SET for active learning
def confidenceLevelsVideoSetDefault(videos_path,nmodel):
    # Get the list of names of the video files
    classes = ['enthusiastic','neutral','concerned']
    img_height = 90
    img_width = 90
    videosList = os.listdir(videos_path) # Lists all files (and directories) in the folder
    confidence_videoset = []

    for video in videosList:
        if os.path.isfile(os.path.join(videos_path, video)): # Considers files only
            vp.extractFaceImages(video, videos_path)
    for video in videosList:
        if os.path.isfile(os.path.join(videos_path, video)):
            #extractFaceImages(video, videos_path)
            face_images = vp.loadFaceImages(video,videos_path)
            labels, confidence = vp.predictVideo(face_images,nmodel,img_height,img_width, classes, models[nmodel]) # Run the model
            confidence_videoset.append([video]+confidence) # Store the mean and standard deviation of the per-frame confidence levels, representative of confidence of full video
    
    sorted(confidence_videoset, key=lambda x: x[1]) # Sort them per ascending mean value
    # The video(s) with the lowest confidence level are considered relevant for the model
    # (and hence will later be annotated and the model re-trained)

    for video in confidence_videoset: # Print the confidence levels of all videos, in ascending order
        print str(video[0])+" -- "+"mean cnf: "+str(video[1])+", stddev conf: "+str(video[2])

class Model(object):

    def __init__(self, data_path, list_classes):
        self._data_path = data_path
        self._list_classes = list_classes

        try:
            # Create all (automatically created) required folders
            os.makedirs(data_path+'classes/')
            os.makedirs(data_path+'models/')
            for label in list_classes: # Create a subfolder per each of the facial expressions
                os.makedirs(data_path+'classes/'+label+'/')
            # Check if the (manually created) required folders exist
            if not os.path.exists(data_path+'videos/') or not os.path.exists(data_path+'annotations/'):
                print 'WARNING: required folders were not found! (videos/ and/or annotations/)'
        except OSError:
            print 'WARNING: there was an error when creating the file tree!'
            pass

    @property
    def data_path(self):
        return self._data_path

    @property
    def list_classes(self):
        return self._list_classes

    ''''verifyVideoSetQuality: run the quality verification on all videos contained inside the videos path'''
    def verifyVideoSetQuality(self):
        qver.runQualityVerification(self._data_path+'videos/',self._data_path+'discarded_videos.txt')
        vproc.deleteProcessData(self._data_path+'videos/')

    '''verifyVideoQuality: run the quality verification on a single specific video, given by its absolute path
    (note that the folder containing the video must only contain this video or other .mp4 videos!)'''
    def verifyVideoQuality(self, video_path, video_file):
        os.system('./faceDetectorExtraction.sh '+video_file+' '+video_path)
        video_name = os.path.splitext(video_file)[0]
        videoPassesTest, cause = qver.verifyQuality(video_path, video_name)
        if videoPassesTest!=True:
            print 'The video passed the quality verification test'
        else:
            print 'The video did not pass the quality verification test -- reason: '+cause
        vproc.deleteProcessData(video_path)

    '''videoSetProcessing: process all videos to extract the face images,
    order them per class for the training and perform image pre-processing on them'''
    def videoSetProcessing(self, annot_type):
        if annot_type==0: # MMI-based annotations format
            vproc.videoProcessorMMI(self._data_path, self._list_classes)
        elif annot_type==1: # TT-based annotations format
            vproc.videoProcessor(self._data_path, self._list_classes)
        else:
            print 'WARNING: wrong annotation type code, videos could not be processed'
        vproc.deleteProcessData(self._data_path+'videos/')

    def trainCNN(self, eval_type, valid_size, nfolds):
        pass


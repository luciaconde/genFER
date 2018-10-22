import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import csv

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def predictVideo(images, nmodel, img_height, img_width):
    '''Inputs each extracted aligned face image (from each frame of the video) into the trained
    neural network and store its prediction.
    Outputs the list of all predictions in order of frames.'''
    image_size=90
    classes = ['enthusiastic','neutral','concerned']
    num_classes = len(classes)
    predict_cls = []
    num_channels=1

    confidence = []
    #print 'Shape of images: '+str(images.shape)

    images = np.array(images)
    batch_size = 32
    num_batches = len(images)/batch_size
    if len(images)%batch_size != 0:
        num_batches += 1
    
    # Reshape the images to fit the format of the network input [num_images image_size image_size num_channels]
    x_total = images.reshape(len(images), img_height,img_width,num_channels)

    batch_count = 0
    for batch in range(num_batches):
        x_batch = x_total[batch_count*batch_size:(batch_count+1)*batch_size-1]

        ## Restore the saved model of the neural network
        sess = tf.Session()
        # Import the network graph
        saver = tf.train.import_meta_graph('face-exp-model{}.meta'.format(nmodel))
        # Load the stored weights
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Accessing the imported default graph
        graph = tf.get_default_graph()

        # Load the tensor containing the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        ## Feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0") 
        y_true = graph.get_tensor_by_name("y_true:0") 
        y_test_images = np.zeros((1, num_classes)) # (Note that each element must have number_classes zeros to fit the labels!)


        # Feed the images and the ground truth to the network needed to obtain y_pred 
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        # result is an array with the probabilities of each image of being each of the classes

        ### Performance measures
        # Convert the number-identified labels from the predictions to actual text labels
        for cnn_output in result:
            predict_cls.append(classes[cnn_output.argmax()])
            level_cnf = max(cnn_output)-min(cnn_output)
            confidence.append(level_cnf) # Save levels of confidence per frame
    
        batch_count += 1

    # Calculate overall level of confidence of the prediction
    confidence_video = [np.mean(confidence), np.std(confidence)]
    print 'Mean/stddev of overall confidence: '+str(confidence_video)

    return predict_cls, confidence_video

def extractFaceImages(video,videos_path):
    '''Extract all the aligned face images from the video frames through OpenFace
    and store them for classification.'''
    # Process the video through OpenFace to extract the aligned face images
    os.system('./faceDetectorExtraction.sh '+video+' '+videos_path)

def loadFaceImages(video,videos_path): # videos_path should be data/test_videos/ for now
    '''Load and process all the aligned face images to prepare them for classification.'''
    image_size = 90
    # Extract name of video
    video_name = os.path.splitext(video)[0]
    # Access the folder containing the extracted face bounding box images
    frames_folder = video_name+'_aligned/'
    frames_path = videos_path+frames_folder
    # Get the list of names of the frames image files
    frames_list = os.listdir(frames_path)
    frames_list.sort()

    images = []

    for frame in frames_list:
        frame_file = frames_path + frame
        #print "File name: "+str(frame_file)
        image = cv2.imread(frame_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        #image = image[0:image_size/3, 0:image_size] # for eyes region only!
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
    images = np.array(images)

    return images

def cleanLabels(labels):
    '''Clean the predicted labels from all potentially spurious predictions.'''
    i = 0
    clean_labels = labels
    for label in labels[1:-1]: # Eliminate any spurious prediction 
        #-- When a frame between two identically tagged frames
        #-- is tagged with a different label, that frame label
        #-- is changed to the one of the surrounding frames
        if labels[i] == labels[i+2] and label != labels[i]:
            clean_labels[i+1] = labels[i]
        #-- When transitioning from one expression to another, if there is a misidentified
        #-- label inbetween (a frame label is different from both of the surrounding frame labels
        #-- plus both surrounding labels are different from each other),
        #-- change it to the label right before
        elif labels[i] != labels[i+2] and label != labels[i] and label != labels[i+2]:
            clean_labels[i+1] = labels[i]
        i += 1
    return clean_labels

def savePredictionsPerFrame(labels, filename):
    '''Store the individual frames' predicted labels in a CSV file.'''
    f = open(filename, "wt")
    check_msg = 'The per-frame labels could not be stored!'
    try:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label]) # Write each predicted label on a new line
        check_msg = 'Per-frame labels were stored.'
    finally:
        f.close()
    return check_msg

def savePredictionsPerExpression(labels, filename):
    '''Store the predicted labels in the same format as the training annotations
    (each row indicates the starting frame, ending frame and predicted expression
    for that range of frames).'''
    f = open(filename, "wt")
    frame_start = 0
    i = 0
    check_msg = 'The per-expression labels could not be stored!'
    try:
        writer = csv.writer(f)
        for label in labels[0:-1]: # Group identically tagged consecutive frames on the same annotated line
            if labels[i+1] != labels[i] or (i+2) >= len(labels):
                writer.writerow([frame_start,i,labels[i]])
                frame_start = (i+1)
            i += 1
        check_msg = 'Per-expression labels were stored.'
    finally:
        f.close()
    return check_msg

def playLabeledVideo(video_path, labels):
    '''Play the test video with the predicted facial expression labels printed on each frame.'''
    # Read video file
    cap = cv2.VideoCapture(video_path)
    print "Path of test video: "+str(video_path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frame_count = 0
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        try: # Display the predicted label of each of the frames
            cv2.putText(frame, labels[frame_count],(50, 50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.4,(0,0,255))
        except IndexError:
            pass
        frame_count += 1
        if ret == True:
 
            # Display the resulting frame
            cv2.imshow('Frame',frame)
 
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
 
        # Break the loop when the end of the video is reached
        else: 
            break
 
    # Release the video capture object when everything's done
    cap.release()
 
    # Close all the frames
    cv2.destroyAllWindows()

## VIDEO PREDICTOR: PER-FRAME FACIAL EXPRESSION CLASSIFIER
def videoPredictor(video_name):
    #video_name = 'myrecording4.mp4'
    #video_name = 'user_response_3549556.mp4'
    video_title = os.path.splitext(video_name)[0]
    video_path = 'data/test_videos_TEST/'
    img_height = 90
    img_width = 90
    nmodel = 16

    extractFaceImages(video_name, video_path)

    face_images = loadFaceImages(video_name,video_path)

    labels, confidence = predictVideo(face_images,nmodel,img_height,img_width) # Predict the facial expression label for each of the face images (that is, for each video frame)
#print 'Predicted labels: '+str(labels)
    clean_labels = cleanLabels(labels)
#print 'Cleaned labels: '+str(clean_labels)

    perframe_labels_areStored = savePredictionsPerFrame(clean_labels, video_path + video_title + '_predictions_perframe.csv')
    print perframe_labels_areStored

    perexp_labels_areStored = savePredictionsPerExpression(clean_labels, video_path + video_title + 'predictions_perexp.csv')
    print perexp_labels_areStored

    playLabeledVideo(video_path+video_name, clean_labels)

# CONFIDENCE LEVEL CALCULATION OF VIDEO SET for active learning
def confidenceLevelsVideoSet():
    # Get the list of names of the video files
    videos_path = "data/test_videos/"
    img_height = 90
    img_width = 90
    nmodel = 2
    videosList = os.listdir(videos_path) # Lists all files (and directories) in the folder
    #frames_list.sort()
    confidence_videoset = []

    '''for video in videosList:
        if os.path.isfile(os.path.join(videos_path, video)): # Considers files only
            extractFaceImages(video, videos_path)'''
    for video in videosList:
        if os.path.isfile(os.path.join(videos_path, video)):
            #extractFaceImages(video, videos_path)
            face_images = loadFaceImages(video,videos_path)
            labels, confidence = predictVideo(face_images,nmodel,img_height,img_width) # Run the model
            confidence_videoset.append([video]+confidence) # Store the mean and standard deviation of the per-frame confidence levels, representative of confidence of full video
    
    sorted(confidence_videoset, key=lambda x: x[1]) # Sort them per ascending mean value
    # The video(s) with the lowest confidence level are considered relevant for the model
    # (and hence will later be annotated and the model re-trained)

    for video in confidence_videoset: # Print the confidence levels of all videos, in ascending order
        print str(video[0])+" -- "+"mean cnf: "+str(video[1])+", stddev conf: "+str(video[2])
    

#videoPredictor('myrecording4.mp4')
#videoPredictor('user_response_3500327.mp4')
#videoPredictor('user_response_3549556.mp4')
#videoPredictor('i_user_response_3524837.mp4')
videoPredictor('test_lucia1.mp4')


#confidenceLevelsVideoSet()

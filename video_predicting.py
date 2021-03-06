import tensorflow as tf
import numpy as np
import os,glob,cv2,shutil
import sys,argparse
import csv

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def predictVideo(images, name_model, img_height, img_width, classes, path_model, folder_model):
    '''Inputs each extracted aligned face image (from each frame of the video) into the trained
    neural network and store its prediction.
    Outputs the list of all predictions in order of frames.'''
    image_size=90
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
        saver = tf.train.import_meta_graph(path_model+folder_model+'{}.meta'.format(name_model))
        # Load the stored weights
        saver.restore(sess, tf.train.latest_checkpoint(path_model+folder_model))

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
    
def savePredictionsPercentagesDefault(labels, filename, video_title):
    pred_percent = open(filename,'wt')
    
    c_concerned = 0
    c_enthusiastic = 0
    c_neutral = 0
    
    # Count the number of predicted labels per class
    for label in labels:
        if label == 'concerned':
            c_concerned += 1
        elif label == 'enthusiastic':
            c_enthusiastic += 1
        elif label == 'neutral':
            c_neutral += 1
            
    # Calculate the percentages
    c_concerned = float(c_concerned*100/len(labels))
    c_enthusiastic = float(c_enthusiastic*100/len(labels))
    c_neutral = float(c_neutral*100/len(labels))
    
    pred_percent.write('Percentages of predicted labels in '+video_title+':\n\n')
    
    try:
        pred_percent.write('concerned: '+str(c_concerned)+'%\n')
        pred_percent.write('enthusiastic: '+str(c_enthusiastic)+'%\n')
        pred_percent.write('neutral: '+str(c_neutral)+'%\n')
    finally:
        pred_percent.close()

def playLabeledVideo(video_path, video_file, labels, storeVideo):
    '''Play the test video with the predicted facial expression labels printed on each frame.'''
    # Read video file
    cap = cv2.VideoCapture(video_path+video_file)
    print "Path of test video: "+str(video_path)
    # Store resolution of video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Store name
    video_name = os.path.splitext(video_file)[0]
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frame_count = 0
    #if storeVideo:
        # Store new video with facial expression labels
        #out = cv2.VideoWriter(video_name+'_predicted.avi',cv2.VideoWriter_fourcc(*'MJPG'), 25, (frame_width,frame_height))
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        try: # Display the predicted label of each of the frames
            cv2.putText(frame, labels[frame_count],(50, 50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.4,(0,0,255))
        except IndexError:
            pass
        frame_count += 1
        #if storeVideo:
            # Write the frame into the new video file
            #out.write(frame)
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
    #out.release()
 
    # Close all the frames
    cv2.destroyAllWindows()



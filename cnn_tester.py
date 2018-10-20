import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def testCNN(images, cls, nmodel, img_height, img_width):
    image_size=90
    classes = ['enthusiastic','neutral','concerned']
    num_classes = len(classes)
    predict_cls = []
    num_channels=1

    images = np.array(images)
    
    # Reshape the images to fit the format of the network input [num_images image_size image_size num_channels]
    #x_batch = images.reshape(len(images), image_size,image_size,num_channels) # normal version (non-cropped)
    x_batch = images.reshape(len(images), img_height,img_width,num_channels)
    #x_batch = images

    ## Restore the saved model of the neural network (by default, the last one created) 
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
    for cnn_output in result:
        predict_cls.append(classes[cnn_output.argmax()])

    # Calculate accuracy
    acc = (sklearn.metrics.accuracy_score(cls,predict_cls))*100
    print "Accuracy: "+str(acc)+"%"

    # Calculate confusion matrix
    confMat = sklearn.metrics.confusion_matrix(cls,predict_cls)
    print "Confusion matrix: (elements are C_ij, where i is the true class and j the predicted class)"
    print(classes)
    print(confMat)

    # Calculate precision, recall and F1-score values
    print(sklearn.metrics.classification_report(cls,predict_cls,target_names=classes))

    return predict_cls

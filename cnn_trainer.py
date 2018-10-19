import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import cnn_tester

# Add seed for consistent random initialization
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Define data for k-fold cross-validation
#nfolds = 10
total_true_labels = []
total_predict_labels = []
# Define data for subject-based validation
nfolds = 7 # number of subjects (people)

batch_size = 64

# Prepare input data
classes = ['positive','neutral','negative']
num_classes = len(classes)

# Define the percentage of the data that will automatically be used for validation
validation_size = 0.1
img_size = 90
num_channels = 1

train_path='data/classes/'

# Load training and validation images, dividing them in folds for cross-validation
#data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size, nfolds=nfolds)
# Load training and validation images, dividing them in subject-based folds for cross-validation
# (each of the folds has all the data corresponding to the same subject -person-)
data = dataset.read_train_sets_persubject(train_path, img_size, classes, validation_size=validation_size, nsubjects=nfolds)


# Define parameters of neural network model
own_filter_size_conv1 = 7 
own_num_filters_conv1 = 32

own_filter_size_conv2 = 1
own_num_filters_conv2 = 64

own_filter_size_conv3 = 3
own_num_filters_conv3 = 96

own_filter_size_conv4 = 5
own_num_filters_conv4 = 256

own_fc_layer_size = 256
own_fc_layer_size2 = 90

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters,
               conv_stride,
               pool_filter_size,
               pool_stride):  
    
    # Initialize the weights and biases (taking the values from a normal distribution) to be trained
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    ## Create the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, conv_stride, conv_stride, 1],
                     padding='SAME')

    layer += biases

    ## Implement max-pooling  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, pool_filter_size, pool_filter_size, 1],
                            strides=[1, pool_stride, pool_stride, 1],
                            padding='SAME')
    ## Feed the output of the max pooling to a ReLu activation function
    layer = tf.nn.relu(layer)

    return layer



def create_flatten_layer(layer):
    # Get shape of previous layer
    layer_shape = layer.get_shape()
    ## Calculate number of features
    num_features = layer_shape[1:4].num_elements()
    ## Flatten the layer (length = number of features)
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    # Define trainable weights and biases
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Generate output (wx+b, where x is the input, w the weights and b the biases)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Start cross-validation
for j in range(nfolds):

    session = tf.Session()
    # (Note that 'None' allows the loading of any number of images)
    #x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x') # normal version (non-cropped)
    x = tf.placeholder(tf.float32, shape=[None, img_size/3,img_size,num_channels], name='x')

    ## Create variable for ground truth labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    # Create CNN
    layer_conv1 = create_convolutional_layer(input=x,
                   num_input_channels=num_channels,
                   conv_filter_size=own_filter_size_conv1,
                   num_filters=own_num_filters_conv1,
                   conv_stride=1,
                   pool_filter_size=2,
                   pool_stride=2)
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                   num_input_channels=own_num_filters_conv1,
                   conv_filter_size=own_filter_size_conv2,
                   num_filters=own_num_filters_conv2,
                   conv_stride=2,
                   pool_filter_size=2,
                   pool_stride=2)

    layer_conv3 = create_convolutional_layer(input=layer_conv2,
                   num_input_channels=own_num_filters_conv2,
                   conv_filter_size=own_filter_size_conv3,
                   num_filters=own_num_filters_conv3,
                   conv_stride=1,
                   pool_filter_size=2,
                   pool_stride=2)

    layer_conv4 = create_convolutional_layer(input=layer_conv3,
                   num_input_channels=own_num_filters_conv3,
                   conv_filter_size=own_filter_size_conv4,
                   num_filters=own_num_filters_conv4,
                   conv_stride=1,
                   pool_filter_size=2,
                   pool_stride=2)
              
    layer_flat = create_flatten_layer(layer_conv4)

    layer_fc1 = create_fc_layer(input=layer_flat,
                         num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                         num_outputs=own_fc_layer_size,
                         use_relu=True)

    layer_fc1_do = tf.nn.dropout(layer_fc1, keep_prob=0.5)

    layer_fc2 = create_fc_layer(input=layer_fc1_do,
                         num_inputs=own_fc_layer_size,
                         num_outputs=own_fc_layer_size2,
                         use_relu=True)

    layer_fc3 = create_fc_layer(input=layer_fc2,
                         num_inputs=own_fc_layer_size2,
                         num_outputs=num_classes,
                         use_relu=False)

    # Create variable for predicted labels
    y_pred = tf.nn.softmax(layer_fc3,name='y_pred')
    y_pred_cls = tf.argmax(y_pred, axis=1)
    session.run(tf.global_variables_initializer())

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,
                                                        labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    session.run(tf.global_variables_initializer())
    
    data.train.setCurrentDataset(nfolds, j)
    data.valid.setCurrentDataset(nfolds, j)

    def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training accuracy: {1:>6.1%}, validation accuracy: {2:>6.1%}, validation loss: {3:.3f}"
        print msg.format(epoch + 1, acc, val_acc, val_loss)

    total_iterations = 0

    saver = tf.train.Saver()
    def train(num_iteration):
        global total_iterations
        
        for i in range(total_iterations,
                       total_iterations + num_iteration):
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size, j)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size, j)

            feed_dict_tr = {x: x_batch,
                               y_true: y_true_batch}
            feed_dict_val = {x: x_valid_batch,
                                  y_true: y_valid_batch}

            session.run(optimizer, feed_dict=feed_dict_tr)

            if i % int(data.train.num_examples[j]/batch_size) == 0: 
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / int(data.train.num_examples[j]/batch_size))    
                
                show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                saver.save(session, './face-exp-model{}'.format(j)) 


        total_iterations += num_iteration
    print "STARTING NEW TRAINING --- TEST FOLD: {} OUT OF {}".format(j+1,nfolds)
    
    train(num_iteration=1000)
    print "Finished training for test fold",format(j)

    # Store the current test dataset (the fold not used during training)
    test_data = []
    test_data.extend(data.train.images[j])
    test_data.extend(data.valid.images[j])
    
    test_labels = []
    test_labels.extend(data.train.cls[j])
    test_labels.extend(data.valid.cls[j])
    
    # Test the trained CNN with the test data fold
    predicted_labels = cnn_tester.testCNN(test_data, test_labels, j, img_size, img_size) # normal version (non-cropped)
    #predicted_labels = cnn_tester.testCNN(test_data, test_labels, j, img_size/3, img_size)

    total_true_labels.extend(test_labels)
    total_predict_labels.extend(predicted_labels)

    # Reset the graph for the new CNN model
    tf.reset_default_graph()

# Print final performance measurements after the k-fold cross-validation
confMat = sklearn.metrics.confusion_matrix(total_true_labels,total_predict_labels)
print "Confusion matrix: (elements are C_ij, where i is the true class and j the predicted class)"
print classes
print confMat

print sklearn.metrics.classification_report(total_true_labels, total_predict_labels, target_names = classes)

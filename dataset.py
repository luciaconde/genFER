import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

from sklearn.model_selection import KFold


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print 'Reading training images'
    for fields in classes:   
        index = classes.index(fields)
        print 'Starting to read {} files (Index: {})'.format(fields, index)
        # The dataset images are stored in subfolders named with the corresponding class label
        # therefore the subfolder name determines the class of each of the loaded images
        path = os.path.join(train_path, fields, '*.bmp')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

def load_train_cropped(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print 'Reading training images'
    for fields in classes:   
        index = classes.index(fields)
        print 'Starting to read {} files (Index: {})'.format(fields, index)
        # The dataset images are stored in subfolders named with the corresponding class label
        # therefore the subfolder name determines the class of each of the loaded images
        path = os.path.join(train_path, fields, '*.bmp')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image[0:image_size/3, 0:image_size] # Crop the upper third of the image
                                                # (the area containing the eyes and eyebrows)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    # Note that the arguments have been shaped as (n,?) to divide them on the n train-val sets
    self._num_examples = (len(cls)-2)*len(cls[0])+len(cls[-1]) # All n-1 complete sets plus the last one (possibly not completed)

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0
    self._current_train_img = []
    self._current_labels = []
    self._current_img_names = []
    self._current_cls = []

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  @property
  def current_train_img(self):
    return self._current_train_img

  @property
  def current_labels(self):
    return self._current_labels

  @property
  def current_img_names(self):
    return self._current_img_names

  @property
  def current_cls(self):
    return self._current_cls

    # Create the current set of images to be used for training
    # (complete set minus the current fold for testing)
  def setCurrentDataset(self, nfold, currentfold):
      self._current_train_img = []
      self._current_labels = []
      self._current_img_names = []
      self._current_cls = []
      
      for i in range(nfold):
          if i!=currentfold: # Only append the data folds that are for training
              self._current_train_img.extend(self._images[i])
              self._current_labels.extend(self._labels[i])
              self._current_img_names.extend(self._img_names[i])
              self._current_cls.extend(self._cls[i])

      print "Size of current dataset: {}".format(len(self._current_cls))

  def next_batch(self, batch_size):
    # Return the next n examples (n=batch_size) from the dataset
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples: # If the batch has been completed
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._current_train_img[start:end], self._current_labels[start:end], self._current_img_names[start:end], self._current_cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size, nfolds):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  fold_size = int(len(cls)/nfolds)

  validation_images = []
  validation_labels = []
  validation_img_names = []
  validation_cls = []

  if isinstance(validation_size, float):
        # number of images for the validation subfold
        validation_size = int(validation_size * fold_size)

  print "fold size: {}, val size: {}".format(fold_size,validation_size)
  train_images = []
  train_labels = []
  train_img_names = []
  train_cls = []

  # the train_ and validation_ data are now divided into (nfold) folds as a matrix
  # each row corresponding to a fold
  for i in range(nfolds):
      print "Saving fold no. {}".format(i)
      init_index = i*fold_size
      end_index = i*fold_size+(fold_size-1)
      print "i_index: {}, f_index: {}".format(init_index,end_index)
      current_images = images[init_index:end_index]
      current_labels = labels[init_index:end_index]
      current_img_names = img_names[init_index:end_index]
      current_cls = cls[init_index:end_index]
      
      validation_images.append(current_images[:validation_size])
      validation_labels.append(current_labels[:validation_size])
      validation_img_names.append(current_img_names[:validation_size])
      validation_cls.append(current_cls[:validation_size])

      train_images.append(current_images[validation_size:])
      train_labels.append(current_labels[validation_size:])
      train_img_names.append(current_img_names[validation_size:])
      train_cls.append(current_cls[validation_size:])

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
  print "num examples train: {}".format(data_sets.train.num_examples)
  print "num examples valid: {}".format(data_sets.valid.num_examples)
      
  return data_sets



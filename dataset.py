import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

# Identifiers for cross-validation methods
VALID_METHOD_KFOLD = 0
VALID_METHOD_SUBJECT = 1

# Identifiers for type of input content
INPUT_FULLFACE = 0
INPUT_EYES = 1

# Identifiers for type of input dataset
DATA_MMI = 0
DATA_TT = 1



def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    num_channels = 1
    img_height = 90
    img_width = 90

    print 'Reading training images'
    for fields in classes:   
        index = classes.index(fields)
        print 'Starting to read {} files (Index: {})'.format(fields, index)
        # The dataset images are stored in subfolders named with the corresponding class label
        # therefore the subfolder name determines the class of each of the loaded images
        path = os.path.join(train_path, fields, '*.bmp')
        files = sorted(glob.glob(path))
        #files.sort(key=lambda x: os.path.basename(x).split('.')[0])
        for fl in files:
            image = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_height, img_width),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)

    img_names,images,labels,cls = (list(t) for t in zip(*sorted(zip(img_names,images,labels,cls))))
    images = np.array(images)
    images = images.reshape(len(images), img_height,img_width,num_channels)
    print 'Shape of images: '+str(images.shape)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

def load_train_cropped(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    num_channels = 1
    img_height = 30
    img_width = 90

    print 'Reading training images'
    for fields in classes:   
        index = classes.index(fields)
        print 'Starting to read {} files (Index: {})'.format(fields, index)
        # The dataset images are stored in subfolders named with the corresponding class label
        # therefore the subfolder name determines the class of each of the loaded images
        path = os.path.join(train_path, fields, '*.bmp')
        files = glob.glob(path)
        #files.sort(key=lambda x: os.path.basename(x).split('.')[0])
        for fl in files:
            image = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)
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
    
    img_names,images,labels,cls = (list(t) for t in zip(*sorted(zip(img_names,images,labels,cls))))
    images = np.array(images)
    images = images.reshape(len(images), img_height,img_width,num_channels)
    print 'Shape of images: '+str(images.shape)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

def calculateNumElem(list_item):
    count = 0
    num_rows = 0
    for i in range(len(list_item)):
        for j in list_item[i]:
            count += 1
    return count

class DataSet(object):

  def __init__(self, images, labels, img_names, cls, valid_method):
    # Note that the arguments have been shaped as (n,?) to divide them on the n train-val sets
    #self._num_examples = (len(cls)-2)*len(cls[0])+len(cls[-1]) # All n-1 complete sets plus the last one (possibly not completed)
    #self._num_examples = calculateNumElem(cls)-1
    #print 'Num examples, way 1: '+str((len(cls)-2)*len(cls[0])+len(cls[-1]))
    #print 'Num examples, way 2: '+str(calculateNumElem(cls))
    self._num_examples = []
    for i in range(len(images)):
        self._num_examples.append(calculateNumElem(cls)-len(images[i]))

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
    self._valid_method = valid_method

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

  @property
  def valid_method(self):
    return self._valid_method

    # Create the current set of images to be used for training
    # (complete set minus the current fold for testing)
  def setCurrentDataset(self, nfold, currentfold):
      self._current_train_img = []
      self._current_labels = []
      self._current_img_names = []
      self._current_cls = []
      #print 'Current fold: '+str(currentfold)+', size of training set: '+str(calculateNumElem(self._current_cls))
      for i in range(nfold):
          if i!=currentfold: # Only append the data folds that are for training
              #print 'Appending fold for training: '+str(i)
              self._current_train_img.extend(self._images[i])
              self._current_labels.extend(self._labels[i])
              self._current_img_names.extend(self._img_names[i])
              self._current_cls.extend(self._cls[i])
              #print 'Size of current fold: '+str(len(self._cls[i]))
              #print 'Size of training set, currently: '+str(len(self._current_cls))
          else:
              print 'Size of current test fold: '+str(len(self._cls[i]))

      # Shuffle (for subject-based cross-validation)
      if self._valid_method == 1:
          self._current_train_img, self._current_labels, self._current_img_names, self._current_cls = shuffle(self._current_train_img, self._current_labels, self._current_img_names, self._current_cls)
      print "Size of current dataset: {}".format(len(self._current_cls))

  def next_batch(self, batch_size, nfold):
    # Return the next n examples (n=batch_size) from the dataset
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples[nfold]: # If the batch has been completed
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      #print 'Batch size: '+str(batch_size)+', num_examples in fold '+str(nfold)+': '+str(self._num_examples[nfold])
      assert batch_size <= self._num_examples[nfold], 'WARNING: some of the data subsets have less samples than the current size of the batches, please decrease batch_size value'
    end = self._index_in_epoch

    return self._current_train_img[start:end], self._current_labels[start:end], self._current_img_names[start:end], self._current_cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size, nfolds, input_type):
  class DataSets(object):
    pass
  data_sets = DataSets()
  
  if input_type==INPUT_FULLFACE:
      images, labels, img_names, cls = load_train(train_path, image_size, classes) # normal version (non-cropped)
  elif input_type==INPUT_EYES:
      images, labels, img_names, cls = load_train_cropped(train_path, image_size, classes)
  else:
      print 'WARNING: input data type is not valid!'

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

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls, VALID_METHOD_KFOLD)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls, VALID_METHOD_KFOLD)
  print "num examples train: {}".format(data_sets.train.num_examples)
  print "num examples valid: {}".format(data_sets.valid.num_examples)
      
  return data_sets





def read_train_sets_persubject(train_path, image_size, classes, validation_size, nsubjects, input_type, dataset_type):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if input_type==INPUT_FULLFACE:
      images, labels, img_names, cls = load_train(train_path, image_size, classes) # normal version (non-cropped)
  elif input_type==INPUT_EYES:
      images, labels, img_names, cls = load_train_cropped(train_path, image_size, classes)
  else:
      print 'WARNING: input data type is not valid!' 

  validation_images = []
  validation_labels = []
  validation_img_names = []
  validation_cls = []

  #print "fold size: {}, val size: {}".format(fold_size,validation_size)
  train_images = []
  train_labels = []
  train_img_names = []
  train_cls = []

  # the train_ and validation_ data are now divided into (nfold) folds as a matrix
  # each row corresponding to a fold
  #subject = 'a'
  init_index = 0
  end_index = 0

  for i in range(nsubjects):
      if dataset_type == DATA_TT:
          subject = img_names[init_index][0] # TT data
      elif dataset_type == DATA_MMI:
          subject = img_names[init_index][0:4] # MMI data
      else:
          print 'WARNING: type of dataset is not valid!'
          break
      print "Saving fold no. {}".format(i)+" -- subject: "+subject
      for name in img_names[init_index:]:
          if dataset_type == DATA_TT:
              if name[0] != subject or end_index>=(len(labels)-1): # If all the data from the current subject has been checked,
                  break
          elif dataset_type == DATA_MMI:
              if str(name[0:4]) != subject or end_index>=(len(labels)-1):
                  break # Store the index of the last one to store the fold and go to the next subject
          end_index += 1

      print 'Init: '+str(init_index)+', end: '+str(end_index)
      current_images = images[init_index:end_index]
      current_labels = labels[init_index:end_index]
      current_img_names = img_names[init_index:end_index]
      current_cls = cls[init_index:end_index]

      # Shuffle before dividing them into training and validation
      current_images, current_labels, current_img_names, current_cls = shuffle(current_images, current_labels, current_img_names, current_cls) 

      validation_samples = int(validation_size*(end_index-init_index))
      
      validation_images.append(current_images[:validation_samples])
      validation_labels.append(current_labels[:validation_samples])
      validation_img_names.append(current_img_names[:validation_samples])
      validation_cls.append(current_cls[:validation_samples])

      train_images.append(current_images[validation_samples:])
      train_labels.append(current_labels[validation_samples:])
      train_img_names.append(current_img_names[validation_samples:])
      train_cls.append(current_cls[validation_samples:])

      print 'Current size train: '+str(calculateNumElem(train_labels))
      #print 'Current size total: '+str(len(train_images[i])+len(validation_images[i]))
      print 'Current size test: '+str(calculateNumElem(validation_labels))

      #subject = chr(ord(subject) + 1) # Go to next 
      init_index = end_index+1
      end_index += 1
      if init_index >= (len(labels)-1):
          break

  '''print 'FOLD NUMBER '+str(i)+' -- included images:'
  print 'TRAINING: '+str(train_img_names)
  print 'VALIDATION: '+str(validation_img_names)'''

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls, VALID_METHOD_SUBJECT)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls, VALID_METHOD_SUBJECT)
  print "num examples train: {}".format(data_sets.train.num_examples)
  print "num examples valid: {}".format(data_sets.valid.num_examples)
      
  return data_sets


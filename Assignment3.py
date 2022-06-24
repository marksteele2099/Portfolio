"""
Python Code for Assignment 3
CS 6302 Predictive Analytics
11/2/2019

Code by Mark Steele
"""


###############
### IMPORTS ###
###############

import skimage
import os
import numpy as np
import matplotlib.pyplot as plt       # for plotting graphs
import h5py

from os import listdir
from skimage import io
from skimage.util import img_as_ubyte
from sklearn.metrics import accuracy_score

from keras.models import Sequential  # Sequenctial model
from keras.models import load_model

# Dense layers, Activation function, and Dropout
from keras.layers.core import Dense, Activation, Dropout 

from keras.optimizers import Adam  # Optimization algorithm
from keras.layers.normalization import BatchNormalization  # Normalization
from keras.layers.convolutional import Conv2D  # Convolutional layers
from keras.layers.convolutional import MaxPooling2D  # Pooling layer
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from prettytable import PrettyTable  # Makes nice looking ASCII tables


#################
### CONSTANTS ###
#################

NUM_ROTATE = 2  # Number of rotations to make
NB_CLASSES = 1  # number of classes (1 indicates binary classes labeled 0/1)
NB_EPOCH = 100
BATCH_SIZE = 50
VERBOSE = 1  # 1 for yes 0 for no
VALIDATION_SPLIT = 0.2
METRICS =['accuracy']
LOSS = 'binary_crossentropy'
IMG_ROWS=32
IMG_COLS=32
IMG_CHANNELS = 3 #color rgb
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)

NUM_KERNELS = 32
KERNAL_INIT = 'glorot_uniform'
KERN_SIZE = (3,3)
POOL_SIZE = (2,2)

NUM_NODES = 100
MONITOR = 'val_loss'
PATIENCE = 8
BEST_ONLY = True
PADDING = "valid"
HIDDEN_ACT = "relu"
OUTPUT_ACT = "sigmoid"

PATH = 'C:/Users/sykom/Desktop/Stuff/College Courses/' + \
       'CS 6302 Predictive Analytics/Assignments/' + \
       'Assignment 03/'
# os.chdir(PATH)  # might make PATH prefixes unnecessary but not recommended
       
train_label_file          = PATH + 'train_class_labels.txt'
train_image_dir           = PATH + 'train/'
augmented_train_folder    = PATH + 'augmented/train/'
new_train_label_file      = PATH + 'augmented/train_class_labels.txt'

test_label_file           = PATH + 'test_class_labels.txt'
test_image_dir            = PATH + 'test/'

saved_data_file           = PATH + 'augmented_cifar_data.h5'
saved_model_dir           = PATH + 'model/'

model_05_original_file    = saved_model_dir + 'original_05.hdf5'
model_001_original_file   = saved_model_dir + 'original_001.hdf5'
model_0005_original_file  = saved_model_dir + 'original_0005.hdf5'
model_05_augmented_file   = saved_model_dir + 'augmented_05.hdf5'
model_001_augmented_file  = saved_model_dir + 'augmented_001.hdf5'
model_0005_augmented_file = saved_model_dir + 'augmented_0005.hdf5'


#################
### FUNCTIONS ###
#################

def get_images(label_path, image_path, name="dataset"):
    """
    Function to retrieve image datasets.
    :param label_path: file path string for dataset labels location
    :param image_path: file path string for dataset images location
    :param name: optional string for naming the dataset
    :returns: numpy arrays of images and their labels, respectively
    """
    labels = {}

    with open(label_path) as f:
        for line in f.read().splitlines():
            k,v = line.split("\t")
            labels[k] = v

    print('number of {0} images: '.format(name), len(labels.keys()))

    files = listdir(image_path)

    #creating numpy matrix/tensor to store the images
    I = io.imread(image_path + files[0])
    x = np.empty(shape = \
                     (len(files), \
                      I.shape[0], \
                      I.shape[1], \
                      I.shape[2]), \
                 dtype = np.int)

    #creating numpy matrix to store the labels
    y = np.empty(shape=(len(files)), dtype = np.int)
    print(x.shape)
    print(y.shape)
    for i in range(0,len(files)):
        if(i%1000 == 0):
            print('done processing ' + str(i) + ' {0} images'.format(name))
        I = io.imread(image_path + files[i])
        x[i, : , : , : ] = I
        y[i] = int(labels[files[i]])
    
    return x, y


def augment_images(label_path, image_path, aug_path, new_path, name, rots):
    """
    Function to artificially increase the size of image datasets.
    :param label_path: file path string for dataset labels location
    :param image_path: file path string for dataset images location
    :param aug_path: file path string for augmented dataset images location
    :param new_path: file path string for augmented dataset labels location
    :param name: optional string for naming the dataset
    :param rots: number of versions of rotated images to create
    :returns: numpy arrays of images and their labels, respectively
    """
    labels = {}

    with open(label_path) as f:
        for line in f.read().splitlines():
            k,v = line.split("\t")
            labels[k] = v

    print('number of {0} images: '.format(name), len(labels.keys()))

    #makes a directory
    files = listdir(image_path)
    os.makedirs(aug_path)

    new_labels = {}
    for i in range(0, len(files)):
        if(i%1000 == 0):
            print('done processing ' + str(i) + ' {0} images'.format(name))
        I = io.imread(image_path + files[i])
        new_labels[files[i]] = labels[files[i]]
        io.imsave(aug_path + files[i], I)
        for j in range(0, rots):
            I1 = skimage.transform.rotate(I, \
                                          angle = np.random.uniform(-45, 45))
            #print(I1.shape)
            newFile = "rotated_" + str(j) + "_" + files[i]
            # img_as_ubyte needed to supress conversion warnings
            io.imsave(aug_path + newFile, img_as_ubyte(I1))
            new_labels[newFile] = labels[files[i]]
                                
    with open(new_path, "w") as f:
        for keys in new_labels.keys():
            f.write(keys + "\t" + new_labels[keys] + "\n")
    
    #creating numpy matrix/tensor to store the images
    files = listdir(aug_path)
    x = np.empty(shape = \
                     (len(files), \
                      I.shape[0], \
                      I.shape[1], \
                      I.shape[2]), \
                 dtype = np.int)

    #creating numpy matrix to store the labels
    y = np.empty(shape=(len(files)), dtype = np.int)
    print(x.shape)
    print(y.shape)
    for i in range(0,len(files)):
        if(i%2000 == 0):
            print('done processing ' + str(i) + ' {0} images'.format(name))
        I = io.imread(aug_path + files[i])
        x[i, : , : , : ] = I
        y[i] = int(new_labels[files[i]])
    
    return x, y


def model(x, y, save_path, learn):
    """
    The model to be used for testing each hyperparameter and augmentation.
    :param x: numpy array of images for training the model
    :param y: numpy array of labels for training the model
    :param save_path: file path string for created model location
    :param learn: float of the learning rate to be used
    :returns: keras model
    """
    OPTIMIZER = Adam(lr = learn)

    cnn_model = Sequential()
    cnn_model.add(Conv2D(NUM_KERNELS, \
                          kernel_size = KERN_SIZE, \
                          padding = PADDING, \
                          kernel_initializer = KERNAL_INIT, \
                          input_shape = INPUT_SHAPE))  # , \
                          # data_format = 'channels_last'))
    cnn_model.add(Activation(HIDDEN_ACT))
    cnn_model.add(MaxPooling2D(pool_size = POOL_SIZE))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.2))
    
    cnn_model.add(Conv2D(2*NUM_KERNELS, \
                          kernel_size = KERN_SIZE, \
                          padding = PADDING, \
                          kernel_initializer = KERNAL_INIT, \
                          input_shape = INPUT_SHAPE))  # , \
                          # data_format='channels_last'))
    cnn_model.add(Activation(HIDDEN_ACT))
    cnn_model.add(MaxPooling2D(pool_size = POOL_SIZE))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.3))
    
    cnn_model.add(Flatten())
    cnn_model.add(Dense(NUM_NODES))
    cnn_model.add(Activation(HIDDEN_ACT))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(NB_CLASSES))
    cnn_model.add(Activation(OUTPUT_ACT))
    cnn_model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = METRICS)
    
    print(cnn_model.summary())
    # filepath='cifar_best_cnn_model2.hdf5'
    checkpoint = ModelCheckpoint(save_path, \
                                 monitor = MONITOR, \
                                 verbose = VERBOSE, \
                                 save_best_only = BEST_ONLY)
    
    early_stopping_monitor = EarlyStopping(monitor = MONITOR, \
                                           patience = PATIENCE)

    tuning = cnn_model.fit(x, \
                           y, \
                           batch_size=BATCH_SIZE, \
                           epochs = NB_EPOCH, \
                           verbose = VERBOSE, \
                           validation_split = VALIDATION_SPLIT, \
                           callbacks = [checkpoint, early_stopping_monitor])
    return tuning


def plotHistory(tuning):
    """
    Function for plotting the training history of a model
    :param tuning: keras model whose history os to be plotted
    :returns: Nothing
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(tuning.history['loss'])
    axs[0].plot(tuning.history['val_loss'])
    axs[0].set_title('loss vs epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'vali'], loc='upper left')
    
    axs[1].plot(tuning.history['acc'])
    axs[1].plot(tuning.history['val_acc'])
    axs[1].set_title('accuracy vs epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylim([0.0,1.0])
    axs[1].legend(['train', 'vali'], loc='upper left')
    plt.show(block = False)
    plt.show()
    
    
def GetAccuracy(model,predictors,response):
    """
    Function to calculate accuracy of a model for a given testing dataset
    :param model: keras model to be tested
    :param predictors: numpy array of images for testing the model
    :param response: numpy array of labels for testing the model
    :returns: The accuracy of the model on the given testing dataset
    """
    # y_classes = [np.argmax(y, axis=None, out=None) for y in response]
    pred_class = model.predict(predictors)
    score = accuracy_score(response, pred_class)
    # score = accuracy_score(y_classes,pred_class)
    return score


def main():
    x_train, y_train = get_images(train_label_file, \
                                  train_image_dir, \
                                  "original training set")
    
    x_augmented, y_augmented = augment_images(train_label_file, \
                                              train_image_dir, \
                                              augmented_train_folder, \
                                              new_train_label_file, \
                                              "augmented training set", \
                                              NUM_ROTATE)

    x_test, y_test = get_images(test_label_file, \
                                  test_image_dir, \
                                  "testing set")

    # Save datasets in h5 format
    with h5py.File(saved_data_file, 'w') as hf:
        hf.create_dataset('x_train', data = x_train)
        hf.create_dataset('y_train', data = y_train)
        hf.create_dataset('x_augmented', data = x_augmented)
        hf.create_dataset('y_augmented', data = y_augmented)
        hf.create_dataset('x_test', data = x_test)
        hf.create_dataset('y_test', data = y_test)
        
    # Load datasets
    with h5py.File(saved_data_file, 'r') as hf:
        x_train = np.array(hf['x_train'])
        x_augmented = np.array(hf['x_augmented'])
        x_test = np.array(hf['x_test'])        
        y_train = np.array(hf['y_train'])
        y_augmented = np.array(hf['y_augmented'])
        y_test = np.array(hf['y_test'])
        
    #normalize between 0 and 1
    x_train = x_train/255
    x_augmented = x_augmented/255
    x_test  = x_test/255

    # # Make labels categorical    
    # print('shape of labels before categorical conversion')
    # print(y_train.shape)
    # print(y_augmented)
    # print(y_test.shape)
    # y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    # y_augmented = np_utils.to_categorical(y_augmented, NB_CLASSES)
    # y_test = np_utils.to_categorical(y_test, NB_CLASSES)
    # print('shape of y_train and y_test after categorical conversion')
    # print(y_train.shape)
    # print(y_augmented)
    # print(y_test.shape)
    
    # Create a folder for the saved learning models
    os.makedirs(saved_model_dir)
    
    # Create and train the models
    model_original_05 = model(x_train, \
                              y_train, \
                              model_05_original_file, \
                              .05)
    model_original_001 = model(x_train, \
                               y_train, \
                               model_001_original_file, \
                               .001)
    model_original_0005 = model(x_train, \
                                y_train, \
                                model_0005_original_file, \
                                .0005)
    model_augmented_05 = model(x_augmented, \
                               y_augmented, \
                               model_05_augmented_file, \
                               .05)
    model_augmented_001 = model(x_augmented, \
                                y_augmented, \
                                model_001_augmented_file, \
                                .001)
    model_augmented_0005 = model(x_augmented, \
                                 y_augmented, \
                                 model_0005_augmented_file, \
                                 .0005)
   
    # These are to be commented out once the proceding code is working
    model_original_05 = load_model(model_05_original_file)
    model_original_001 = load_model(model_001_original_file)
    model_original_0005 = load_model(model_0005_original_file)
    model_augmented_05 = load_model(model_05_augmented_file)
    model_augmented_001 = load_model(model_001_augmented_file)
    model_augmented_0005 = load_model(model_0005_augmented_file)
    
    # Calculate accuracies
    acc_org_05 = GetAccuracy(model_original_05, x_test, y_test)
    acc_org_001 = GetAccuracy(model_original_001, x_test, y_test)
    acc_org_0005 = GetAccuracy(model_original_0005, x_test, y_test)
    acc_aug_05 = GetAccuracy(model_augmented_05, x_test, y_test)
    acc_aug_001 = GetAccuracy(model_augmented_001, x_test, y_test)
    acc_aug_0005 = GetAccuracy(model_augmented_0005, x_test, y_test)
    
    table = PrettyTable()
    table.field_names = ["Dataset", "lr = 0.05", "lr = 0.001", "lr = 0.0005"]
    table.add_row(["Original", acc_org_05, acc_org_001, acc_org_0005])
    table.add_row(["Augmented", acc_aug_05, acc_aug_001, acc_aug_0005])
    
    print(table)
    
    # # Plot histories
    # plotHistory(model_original_05)
    # plotHistory(model_original_001)
    # plotHistory(model_original_0005)
    # plotHistory(model_augmented_05)
    # plotHistory(model_augmented_001)
    # plotHistory(model_augmented_0005)
    
############
### MAIN ###
############
    
if __name__ == "__main__":
    main()
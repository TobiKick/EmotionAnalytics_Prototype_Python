#!/usr/bin/env python
# coding: utf-8


############################# IMPORT STATEMENTS ########################################################
#Import Python modules
import time
from datetime import datetime
import numpy as np
from mtcnn import MTCNN
from numpy import asarray
from PIL import Image
import cv2
import os
import json
import csv
from imutils import face_utils 
import argparse 
import imutils 
import dlib 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import math_ops

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator

#Import Keras modules
from keras.layers import Dense, Flatten, Input, Dropout, Conv1D, Conv2D, LSTM, Concatenate, Reshape, MaxPool1D, MaxPool2D, BatchNormalization
from keras import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.utils import np_utils, Sequence
import keras.backend as K
import keras as keras
from menpofit.io import load_fitter
from menpofit.aam import load_balanced_frontal_face_fitter

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


############################# SETUP PROJECT PARAMETERS ########################################################
LOAD_PROGRESS_FROM_MODEL = False
SAVE_PROGRESS_TO_MODEL = True

RUN_LOCAL = False
CONSTRUCT_DATA = False
CROSS_VALIDATION = False

SHUFFLE_FOLD = True
ORIGINAL_IMAGES = False
COMBINED_IMAGES = False
LAYER_REGULARIZATION = True

DATA_AUGMENTATION = True
WITH_LANDMARKS = False
LANDMARKS_ONLY = False


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

FOLD_ARRAY = [0, 1, 2, 3, 4]
FOLD_SIZE = 115 # number of folders/subjects in one fold
BATCH_SIZE = 32

PATH_TO_DATA = 'AFEW-VA'
PATH_TO_EVALUATION = 'AFEW-VA_TEST'
DATA_DIR_PREDICT = ''
IMG_FORMAT = '.png'

EPOCHS = 1000
LEARNING_RATE = 0.01

if RUN_LOCAL == True:
    PATH_TO_DATA = r"C:\Users\Tobias\Desktop\Master-Thesis\Data\AFEW-VA"
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    EPOCHS = 2
    FOLD_ARRAY = [0, 1]
    FOLD_SIZE = 1  # number of folders/subjects in one fold


########################################################################################################
########################################################################################################
########################################################################################################


## setting Keras sessions for each network - First Network
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.keras.backend.set_session(sess)
detector = MTCNN()
p = "shape_predictor_68_face_landmarks.dat"
dlib_predictor = dlib.shape_predictor(p) 


def detect_face(image):
    global sess
    global graph
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        face = detector.detect_faces(image)
        if len(face) >= 1:
            return face
        else:
            print("No face detected")
            return []


def extract_face_from_image(image, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    if ORIGINAL_IMAGES == False:
        face = detect_face(image)
        print(face)
        if face == []:
            face_image = Image.fromarray(image)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)      
            return face_array
        else:
            # extract the bounding box from the requested face
            box = np.asarray(face[0].get("box", ""))
            box[box < 0] = 0
            x1, y1, width, height =  box
            
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face_boundary = image[y1:y2, x1:x2]

            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            return face_array
    else:
        face_image = Image.fromarray(image)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)      
        return face_array


def extract_face_from_image_w_landmarks(image, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
        landmarks = []
        face = detect_face(image)

        if face == []:
            face_image = Image.fromarray(image)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)    
            print(np.array(np.zeros((136,))))  
            return face_array, np.array(np.zeros((136,)))
        else:            
            # extract the bounding box from the requested face
            box = np.asarray(face[0].get("box", ""))
            box[box < 0] = 0
            x1, y1, width, height =  box
            x2, y2 = x1 + width, y1 + height

            rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
            landmarks = dlib_predictor(image, rect)
            np_landmarks = []
            for n in range(0,68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                np_landmarks.append([x, y])
            np_landmarks = np.array(np_landmarks)
            np_landmarks_flatten = np_landmarks.flatten()

            if ORIGINAL_IMAGES == False:
                # extract the face
                face_boundary = image[y1:y2, x1:x2]
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize(required_size)
                face_array = asarray(face_image)
                return face_array, np_landmarks_flatten
            else:
                face_image = Image.fromarray(image)
                face_image = face_image.resize(required_size)
                face_array = asarray(face_image)      
                return face_array, np_landmarks_flatten


def get_labels_from_file(path_to_file, folder):
    filenames = []
    labels = []
    
    with open(path_to_file) as p:
        data = json.load(p)
        
    if not 'frames' in data or len(data['frames']) == 0:     
        exit(0)
    else:
        frames = data['frames']
        for key, value in frames.items():
            filenames.append(str(folder + '/' + key + IMG_FORMAT))
            labels.append([value['valence'], value['arousal']])
    
    return filenames, labels


def constructing_data_list_eval(root_data_dir):
    filenames = []
    labels = []
    
    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    f, l = get_labels_from_file(os.path.join(root_data_dir, train_dir, file), train_dir)
                    filenames.extend(f)
                    labels.extend(l)

    print(len(filenames))
    return np.array(filenames), np.array(labels)


def constructing_data_list(root_data_dir, fold_size):
    filenames = []
    labels = []
    filenames_list = []
    labels_list = []
    i = 1

    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    f, l = get_labels_from_file(os.path.join(root_data_dir, train_dir, file), train_dir)
                    filenames.extend(f)
                    labels.extend(l)

        if i == fold_size:
            filenames_list.append(np.array(filenames))
            labels_list.append(np.array(labels))
            filenames = []
            labels = []
            i = 0

        i = i + 1

    return np.array(filenames_list), np.array(labels_list)


def preloading_data(path_to_data, filenames):
    list_faces = []
    for file_name in filenames:
        img = cv2.imread(os.path.join(path_to_data, str(file_name)))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = extract_face_from_image(img)
            list_faces.append(face)

    return np.array(list_faces)


def preloading_data_w_landmarks(path_to_data, filenames):
    list_faces = []
    list_landmarks = []

    for file_name in filenames:
        img = cv2.imread(os.path.join(path_to_data, str(file_name)))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face, landmarks = extract_face_from_image_w_landmarks(img)
            list_faces.append(face)
            list_landmarks.append(landmarks)

    return np.array(list_faces), np.array(list_landmarks)


def custom_vgg_model(is_trainable):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(sess2)
        
        if COMBINED_IMAGES == False and LAYER_REGULARIZATION == False:
            model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            for layer in model_VGGFace.layers:
                layer.trainable = is_trainable

            last_layer = model_VGGFace.get_layer('avg_pool').output    
            x = Flatten(name='flatten')(last_layer)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = BatchNormalization()(x)
            out1 = Dense(21, activation='softmax', name='out1')(x)
            out2 = Dense(21, activation='softmax', name='out2')(x)
            custom_vgg_model = Model(inputs= model_VGGFace.input, outputs= [out1, out2])
            return custom_vgg_model


        elif COMBINED_IMAGES == False and LAYER_REGULARIZATION == True:
            model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

            regularizer = keras.regularizers.l2(0.01)
            if is_trainable == False:
                for layer in model_VGGFace.layers:
                    layer.trainable = False
            else:
                for layer in model_VGGFace.layers:
                    layer.trainable = True
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)
                model_VGGFace.save_weights("model_checkpoints/VGGFace_Regularization.h5")
                model_json = model_VGGFace.to_json()
                model_VGGFace = keras.models.model_from_json(model_json)
                model_VGGFace.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)

            last_layer = model_VGGFace.get_layer('avg_pool').output    
            x = Flatten(name='flatten')(last_layer)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = BatchNormalization()(x)
            out1 = Dense(21, activation='softmax', name='out1')(x)
            out2 = Dense(21, activation='softmax', name='out2')(x)
            custom_vgg_model = Model(inputs= model_VGGFace.input, outputs= [out1, out2])
            return custom_vgg_model


        elif COMBINED_IMAGES == True and LAYER_REGULARIZATION == False:
            return 0


        else:
            model_VGGFace_1 = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            model_VGGFace_2 = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

            regularizer = keras.regularizers.l2(0.01)
            if is_trainable == False:
                for layer in model_VGGFace_1.layers:
                    layer.trainable = False
                for layer in model_VGGFace_2.layers:
                    layer.trainable = False
            else:
                for layer in model_VGGFace_1.layers:
                    layer.trainable = True
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)
                model_VGGFace_1.save_weights("model_checkpoints/VGGFace_Regularization.h5")
                model_json = model_VGGFace_1.to_json()
                model_VGGFace_1 = keras.models.model_from_json(model_json)
                model_VGGFace_1.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)

                for layer in model_VGGFace_2.layers:
                    layer.trainable = True
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)
                model_VGGFace_2.save_weights("model_checkpoints/VGGFace_Regularization.h5")
                model_json = model_VGGFace_2.to_json()
                model_VGGFace_2 = keras.models.model_from_json(model_json)
                model_VGGFace_2.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)

            for layer in model_VGGFace_2.layers:
                layer.name = str(layer.name) + "_2"

            last_layer_1 = model_VGGFace_1.get_layer('avg_pool').output
            x_1 = Flatten(name='flatten')(last_layer_1)
            last_layer_2 = model_VGGFace_2.get_layer('avg_pool_2').output  
            x_2 = Flatten(name='flatten_2')(last_layer_2)  
            x = Concatenate(axis=-1)([x_1, x_2])

            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = BatchNormalization()(x)
            out1 = Dense(21, activation='softmax', name='out1')(x)
            out2 = Dense(21, activation='softmax', name='out2')(x)
            custom_vgg_model = Model(inputs= [model_VGGFace_1.input, model_VGGFace_2.input], outputs= [out1, out2])
            return custom_vgg_model


def custom_vgg_model_w_landmarks(is_trainable):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(sess2)
        
        model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        # model_VGGFace.trainable = True
        # for layer in model_VGGFace.layers[:-10]:   ## all layers except the last .. layers
        #     layer.trainable = False

        for layer in model_VGGFace.layers:
           layer.trainable = is_trainable

        last_layer = model_VGGFace.get_layer('avg_pool').output    
        x = Flatten(name='flatten')(last_layer)
        landmarks_input = Input(shape=(136,))

        x = Concatenate(axis=-1)([x, landmarks_input])
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        out1 = Dense(1, activation='tanh', name='out1')(x)
        out2 = Dense(1, activation='tanh', name='out2')(x)
        custom_vgg_model = Model(inputs= [model_VGGFace.input, landmarks_input] , outputs= [out1, out2])
        return custom_vgg_model

def custom_landmarks_model():
    landmarks_input = Input(shape=(136,))
    
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)

    x = Dense(4096, activation='relu')(landmarks_input)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    out1 = Dense(1, activation='tanh', name='out1')(x)
    out2 = Dense(1, activation='tanh', name='out2')(x)
    custom_landmarks_model = Model(inputs=landmarks_input , outputs= [out1, out2])
    
    return custom_landmarks_model


def custom_model(is_trainable):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(sess2)
 
        model_input = Input(shape=(224, 224, 3))
        x = Conv2D(64, (11, 11), activation='relu')(model_input)
        x = Conv2D(64, (11, 11), activation='relu')(x)
        x = Conv2D(64, (11, 11), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Conv2D(256, (5, 5), activation='relu')(x)
        x = Conv2D(256, (5, 5), activation='relu')(x)
        x = Conv2D(256, (5, 5), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Flatten(name='flatten')(x)    
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        out1 = Dense(1, activation='tanh', name='out1')(x)
        out2 = Dense(1, activation='tanh', name='out2')(x)
        custom_vgg_model = Model(inputs=model_input , outputs= [out1, out2])

        return custom_vgg_model



def one_hot_encoding(input_array):
    values = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # integer encode
    label_encoder = LabelEncoder()
    label_encoder.fit(values)
    values_enc = label_encoder.transform(values)
    input_array_encoding = label_encoder.transform(input_array)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    values_enc = values_enc.reshape(len(values_enc), 1)
    onehot_encoder.fit(values_enc)
    input_array_reshape = input_array_encoding.reshape(len(input_array_encoding), 1)
    input_array_onehot = onehot_encoder.transform(input_array_reshape)
    return input_array_onehot


def one_hot_undo(one_hot_encoded):
    values = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    label_encoder = LabelEncoder()
    label_encoder.fit(values)
    inverted = label_encoder.inverse_transform([argmax(one_hot_encoded)])
    return inverted


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 


def corr(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = tf.math.divide_no_nan(r_num, r_den, name="division")
    #r = r_num / r_den
    return K.mean(r)


def corr_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


class CustomFineTuningCallback(keras.callbacks.Callback):
    def __init__(self):
        super(CustomFineTuningCallback, self).__init__()
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 3:
            for layer in self.model.layers:
                layer.trainable = True
            
            self.model.compile(loss = rmse, optimizer = self.model.optimizer, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            LR = 0.01
            keras.backend.set_value(self.model.optimizer.lr, LR)
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, LR))


def scheduler(epoch, lr):
    print("Learning rate: " + str(lr))
    if epoch == 0:
        return float(lr)
    else:
        return float(lr * 0.90) 

def multi_out(gen):
    for x, y in gen:
        yield x, [y[:,0], y[:,1]]



def run_model():
    if  CONSTRUCT_DATA == True and ORIGINAL_IMAGES == True and SHUFFLE_FOLD == False:
        filenames, labels = constructing_data_list(PATH_TO_DATA, FOLD_SIZE)

        fold_input = []
        fold_input_landmarks = []
        fold_target = []
        for i in FOLD_ARRAY:
            target = np.true_divide(labels[i], 10)
            fold_target.append(target)

            preload_input, preload_landmarks = preloading_data_w_landmarks(PATH_TO_DATA, filenames[i])
            fold_input.append(preload_input)
            fold_input_landmarks.append(preload_landmarks)


        fold_target = np.array(fold_target)
        fold_input = np.array(fold_input)
        fold_input_landmarks = np.array(fold_input_landmarks)
        np.save('numpy/Y_fold_target_original.npy', fold_target)
        np.save('numpy/X_fold_input_original.npy', fold_input)
        np.save('numpy/X_fold_input_landmarks.npy', fold_input_landmarks)
            
        test_files, test_labels = constructing_data_list_eval(PATH_TO_EVALUATION)
        test_input, test_input_landmarks = preloading_data_w_landmarks(PATH_TO_EVALUATION, test_files)
        test_target = np.true_divide(test_labels, 10)
        np.save('numpy/Y_test_target_original.npy', test_target)
        np.save('numpy/X_test_input_original.npy', test_input)
        np.save('numpy/X_test_input_landmarks.npy', test_input_landmarks)
        

    elif CONSTRUCT_DATA == True and ORIGINAL_IMAGES == True and SHUFFLE_FOLD == True:
        filenames, labels = constructing_data_list(PATH_TO_DATA, FOLD_SIZE)

        fold_input = []
        fold_target = []
        for i in FOLD_ARRAY:
            f, l = shuffle(filenames[i], labels[i], random_state=0)

            target = np.true_divide(l, 10)
            fold_target.append(target)

            preload_input = preloading_data(PATH_TO_DATA, f)
            fold_input.append(preload_input)

        fold_target = np.array(fold_target)
        fold_input = np.array(fold_input)
        np.save('numpy/Y_fold_target_shuffled_original.npy', fold_target)
        np.save('numpy/X_fold_input_shuffled_original.npy', fold_input)

        test_files, test_labels = constructing_data_list_eval(PATH_TO_EVALUATION)
        test_input = preloading_data(PATH_TO_EVALUATION, test_files)
        test_target = np.true_divide(test_labels, 5)
        np.save('numpy/X_test_input_shuffled_original.npy', test_input)
        np.save('numpy/Y_test_target_shuffled_original.npy', test_target)

    elif CONSTRUCT_DATA == True and ORIGINAL_IMAGES == False and SHUFFLE_FOLD == False:
        filenames, labels = constructing_data_list(PATH_TO_DATA, FOLD_SIZE)

        fold_input = []
        fold_target = []
        for i in FOLD_ARRAY:
            target = np.true_divide(labels[i], 10)
            fold_target.append(target)

            preload_input = preloading_data(PATH_TO_DATA, filenames[i])
            fold_input.append(preload_input)

        fold_target = np.array(fold_target)
        fold_input = np.array(fold_input)
        np.save('numpy/Y_fold_target.npy', fold_target)
        np.save('numpy/X_fold_input.npy', fold_input)

        test_files, test_labels = constructing_data_list_eval(PATH_TO_EVALUATION)
        test_input = preloading_data(PATH_TO_EVALUATION, test_files)
        test_target = np.true_divide(test_labels, 10)
        np.save('numpy/X_test_input.npy', test_input)
        np.save('numpy/Y_test_target.npy', test_target)
    
    elif CONSTRUCT_DATA == True and ORIGINAL_IMAGES == False and SHUFFLE_FOLD == True:
        filenames, labels = constructing_data_list(PATH_TO_DATA, FOLD_SIZE)

        fold_input = []
        fold_target = []
        for i in FOLD_ARRAY:
            f, l = shuffle(filenames[i], labels[i], random_state=0)

            target_V = one_hot_encoding(l[:,0])
            target_A = one_hot_encoding(l[:,1])
            target = np.zeros((len(target_V), 2, len(target_V[0])))
            for i in range(0, len(target_V)):
                target[i] = [target_V[i], target_A[i]]

            fold_target.append(target)

            preload_input = preloading_data(PATH_TO_DATA, f)
            fold_input.append(preload_input)

        fold_input = np.array(fold_input)
        np.save('numpy/X_fold_input_shuffled.npy', fold_input)
        fold_target = np.array(fold_target)
        np.save('numpy/Y_fold_target_shuffled.npy', fold_target)
        

        test_files, test_labels = constructing_data_list_eval(PATH_TO_EVALUATION)
        test_input = preloading_data(PATH_TO_EVALUATION, test_files)
        np.save('numpy/X_test_input_shuffled.npy', test_input)

        test_target_V = one_hot_encoding(test_labels[:,0])
        test_target_A = one_hot_encoding(test_labels[:,1])
        test_target = np.zeros((len(test_target_V), 2, len(test_target_V[0])))
        for i in range(0, len(test_target_V)):
            test_target[i] = [test_target_V[i], test_target_A[i]]
        np.save('numpy/Y_test_target_shuffled.npy', test_target)


    if CONSTRUCT_DATA == False:
        fold_input_landmarks = np.load('numpy/X_fold_input_landmarks.npy', allow_pickle=True)
        if COMBINED_IMAGES == True:
            fold_input = np.load('numpy/X_fold_input.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target.npy', allow_pickle=True)
            fold_input_2 = np.load('numpy/X_fold_input_original.npy', allow_pickle=True)
            fold_target_2 = np.load('numpy/Y_fold_target_original.npy', allow_pickle=True)

            test_data_input_2 = np.load('numpy/X_test_input_original.npy', allow_pickle=True)
            test_data_input = np.load('numpy/X_test_input.npy', allow_pickle=True)
            test_data_target = np.load('numpy/Y_test_target.npy', allow_pickle=True)

        elif ORIGINAL_IMAGES == True and SHUFFLE_FOLD == False:
            fold_input = np.load('numpy/X_fold_input_original.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target_original.npy', allow_pickle=True)
            test_data_input = np.load('numpy/X_test_input_original.npy', allow_pickle=True)
            test_data_target = np.load('numpy/Y_test_target_original.npy', allow_pickle=True)

        elif ORIGINAL_IMAGES == True and SHUFFLE_FOLD == True:
            fold_input = np.load('numpy/X_fold_input_shuffled_original.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target_shuffled_original.npy', allow_pickle=True)
            test_data_input = np.load('numpy/X_test_input_shuffled_original.npy', allow_pickle=True)
            test_data_target = np.load('numpy/Y_test_target_shuffled_original.npy', allow_pickle=True)#

        elif ORIGINAL_IMAGES == False and SHUFFLE_FOLD == False:
            fold_input = np.load('numpy/X_fold_input.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target.npy', allow_pickle=True)
            test_data_input = np.load('numpy/X_test_input.npy', allow_pickle=True)
            test_data_target = np.load('numpy/Y_test_target.npy', allow_pickle=True)

        elif ORIGINAL_IMAGES == False and SHUFFLE_FOLD == True:
            fold_input = np.load('numpy/X_fold_input_shuffled.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target_shuffled.npy', allow_pickle=True)

            test_input = np.load('numpy/X_test_input_shuffled.npy', allow_pickle=True)
            test_target = np.load('numpy/Y_test_target_shuffled.npy', allow_pickle=True)


    # K-fold Cross Validation model evaluation
    history_accuracy_1 = []
    history_corr_1 = []
    history_rmse_1 = []
    history_accuracy_2 = []
    history_corr_2 = []
    history_rmse_2 = []

    val_history_accuracy_1 = []
    val_history_corr_1 = []
    val_history_rmse_1 = []
    val_history_accuracy_2 = []
    val_history_corr_2 = []
    val_history_rmse_2 = []

    if CROSS_VALIDATION == True:
        for i in FOLD_ARRAY:
            cb_bestModel = ModelCheckpoint('model_checkpoints/model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            cb_earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) # waiting for X consecutive epochs that don't reduce the val_loss
            # cb_learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001, verbose=1)
            cb_learningRate =  LearningRateScheduler(scheduler)

            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            inputs_test = fold_input[i]
            targets_test = fold_target[i]
            inputs_train = []
            targets_train = []

            for t in FOLD_ARRAY:
                if t != i:
                    if inputs_train != []:
                        inputs_train = np.concatenate((fold_input[t], inputs_train), axis=0)
                        targets_train = np.concatenate((fold_target[t], targets_train), axis=0)
                    else:
                        inputs_train = np.array(fold_input[t])
                        targets_train = np.array(fold_target[t])
                        print(inputs_train[t].shape)


            model = custom_vgg_model(False)
            model.summary()
            opt = Adam(learning_rate = 0.1)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), batch_size=BATCH_SIZE, verbose=1, epochs=3)
            model.save_weights("model_checkpoints/model_non_trainable.h5")

            history_accuracy_1.extend(scores.history['out1_accuracy'])
            history_corr_1.extend(scores.history['out1_corr'])
            history_rmse_1.extend(scores.history['out1_rmse'])
            history_accuracy_2.extend(scores.history['out2_accuracy'])
            history_corr_2.extend(scores.history['out2_corr'])
            history_rmse_2.extend(scores.history['out2_rmse'])

            val_history_accuracy_1.extend(scores.history['val_out1_accuracy'])
            val_history_corr_1.extend(scores.history['val_out1_corr'])
            val_history_rmse_1.extend(scores.history['val_out1_rmse'])

            val_history_accuracy_2.extend(scores.history['val_out2_accuracy'])
            val_history_corr_2.extend(scores.history['val_out2_corr'])
            val_history_rmse_2.extend(scores.history['val_out2_rmse'])

            model = custom_vgg_model(True)
            model.load_weights("model_checkpoints/model_non_trainable.h5")
            model.summary()
            opt = Adam(learning_rate = LEARNING_RATE)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop, cb_learningRate])
            
            model.load_weights("model_checkpoints/model_best.h5")
            result = model.evaluate(inputs_test, [targets_test[:,0], targets_test[:,1]], verbose=1, batch_size=BATCH_SIZE)

            with open('CrossValidation.csv', "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(str(i))
                wr.writerow(result)
                

            print("Just wrote to CrossValidation file")
            history_accuracy_1.extend(scores.history['out1_accuracy'])
            history_corr_1.extend(scores.history['out1_corr'])
            history_rmse_1.extend(scores.history['out1_rmse'])
            history_accuracy_2.extend(scores.history['out2_accuracy'])
            history_corr_2.extend(scores.history['out2_corr'])
            history_rmse_2.extend(scores.history['out2_rmse'])

            val_history_accuracy_1.extend(scores.history['val_out1_accuracy'])
            val_history_corr_1.extend(scores.history['val_out1_corr'])
            val_history_rmse_1.extend(scores.history['val_out1_rmse'])

            val_history_accuracy_2.extend(scores.history['val_out2_accuracy'])
            val_history_corr_2.extend(scores.history['val_out2_corr'])
            val_history_rmse_2.extend(scores.history['val_out2_rmse'])
        
        if SAVE_PROGRESS_TO_MODEL:
            model.save_weights("model_checkpoints/model_top.h5")
            print("Saved model to disk")

        with open('CrossValidation.csv', "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(model.metrics_names)

        my_dict = {}
        my_dict['out1_accuracy'] = history_accuracy_1
        my_dict['out1_corr'] = history_corr_1
        my_dict['out1_rmse'] = history_rmse_1
        my_dict['out2_accuracy'] = history_accuracy_2
        my_dict['out2_corr'] = history_corr_2
        my_dict['out2_rmse'] = history_rmse_2

        my_dict['val_out1_accuracy'] = val_history_accuracy_1
        my_dict['val_out1_corr'] = val_history_corr_1
        my_dict['val_out1_rmse'] = val_history_rmse_1
        my_dict['val_out2_accuracy'] = val_history_accuracy_2
        my_dict['val_out2_corr'] = val_history_corr_2
        my_dict['val_out2_rmse'] = val_history_rmse_2
    
    else:
        cb_bestModel = ModelCheckpoint('model_checkpoints/model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        cb_earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) # waiting for X consecutive epochs that don't reduce the val_loss
        # cb_learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001, verbose=1)
        cb_learningRate =  LearningRateScheduler(scheduler)
        
        if COMBINED_IMAGES == True:
            i = 0
            inputs_test = fold_input[i]
            targets_test = fold_target[i]
            inputs_test_2 = fold_input_2[i]
            targets_test_2 = fold_target_2[i]

            inputs_train = []
            targets_train = []
            inputs_train_2 = []
            targets_train_2 = []

            for t in FOLD_ARRAY:
                if t != i:
                    if inputs_train != []:
                        inputs_train = np.concatenate((fold_input[t], inputs_train), axis=0)
                        targets_train = np.concatenate((fold_target[t], targets_train), axis=0)

                        inputs_train_2 = np.concatenate((fold_input_2[t], inputs_train_2), axis=0)
                        targets_train_2 = np.concatenate((fold_target_2[t], targets_train_2), axis=0)
                    else:
                        inputs_train = np.array(fold_input[t])
                        targets_train = np.array(fold_target[t])
                        
                        inputs_train_2 = np.array(fold_input_2[t])
                        targets_train_2 = np.array(fold_target_2[t])
            inputs_train, targets_train, inputs_train_2, targets_train_2 = shuffle(inputs_train, targets_train, inputs_train_2, targets_train_2, random_state=0)
        
        else:
            i = 0
            inputs_test = fold_input[i]
            inputs_test_landmarks = fold_input_landmarks[i]
            targets_test = fold_target[i]
            

            inputs_train = []
            inputs_train_landmarks = []
            targets_train = []

            for t in FOLD_ARRAY:
                if t != i:
                    if inputs_train != []:
                        inputs_train = np.concatenate((fold_input[t], inputs_train), axis=0)
                        targets_train = np.concatenate((fold_target[t], targets_train), axis=0)
                        inputs_train_landmarks = np.concatenate((fold_input_landmarks[t], inputs_train_landmarks), axis=0)
                    else:
                        inputs_train = np.array(fold_input[t])
                        targets_train = np.array(fold_target[t])
                        inputs_train_landmarks = np.array(fold_input_landmarks[t])
            inputs_train, targets_train, inputs_train_landmarks = shuffle(inputs_train, targets_train, inputs_train_landmarks, random_state=0)
            print(inputs_train.shape)
            print(targets_train.shape)
            print(inputs_train_landmarks.shape)


        if DATA_AUGMENTATION == True and COMBINED_IMAGES == False and WITH_LANDMARKS == False:
            datagen = ImageDataGenerator(
                # featurewise_center=True,
                # featurewise_std_normalization=True,
                rotation_range=30,
                width_shift_range=0.25,
                height_shift_range=0.25,
                horizontal_flip=True,
                brightness_range=[0.5, 1.0],
                zoom_range=0.3)

            inputs_train = np.array(inputs_train)            
            datagen.fit(inputs_train)
            gen1 = datagen.flow(inputs_train, targets_train, batch_size=BATCH_SIZE)
            train_steps = len(gen1)
            train = multi_out(gen1)

            model = custom_vgg_model(False)
            model.summary()
            opt = Adam(learning_rate = 0.001)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), validation_steps=(len(inputs_test)/BATCH_SIZE) ,verbose=1, epochs=3, callbacks = [cb_bestModel ,cb_earlyStop])
            model.save_weights("model_checkpoints/model_non_trainable.h5")

            model = custom_vgg_model(True)
            model.load_weights("model_checkpoints/model_non_trainable.h5")
            model.summary()
            opt = Adam(learning_rate = 0.0001)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            
            scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), validation_steps=(len(inputs_test)/BATCH_SIZE) ,verbose=1, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
            model.load_weights("model_checkpoints/model_best.h5")
            result = model.evaluate(test_input, [test_target[:,0], test_target[:,1]], verbose=1, batch_size=BATCH_SIZE)

        elif LANDMARKS_ONLY == True:
            model = custom_landmarks_model()
            model.summary()
            opt = Adam(learning_rate = 0.01)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit(inputs_train_landmarks, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test_landmarks, [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
            
            test_input_landmarks = np.load('numpy/X_test_input_landmarks.npy', allow_pickle=True)
            model.load_weights("model_checkpoints/model_best.h5")
            result = model.evaluate(test_input_landmarks, [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)


        elif COMBINED_IMAGES == True and WITH_LANDMARKS == False:
            model = custom_vgg_model(False)
            model.summary()
            opt = Adam(learning_rate = 0.01)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit([inputs_train, inputs_train_2], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_test, inputs_test_2], [targets_test[:,0], targets_test[:,1]]), batch_size=BATCH_SIZE, verbose=1, epochs=3)
            model.save_weights("model_checkpoints/model_non_trainable.h5")
            
            LEARNING_RATE = 0.01
            model = custom_vgg_model(True)
            model.load_weights("model_checkpoints/model_non_trainable.h5")
            model.summary()
            opt = Adam(learning_rate = LEARNING_RATE)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit([inputs_train, inputs_train_2], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_test, inputs_test_2], [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
            
            model.load_weights("model_checkpoints/model_best.h5")
            result = model.evaluate([test_data_input, test_data_input_2], [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)

        elif COMBINED_IMAGES == False and WITH_LANDMARKS == False:
            model = custom_vgg_model(True)
            model.summary()
            opt = Adam(learning_rate = 0.01)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            
            scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
            model.load_weights("model_checkpoints/model_best.h5")
            result = model.evaluate(test_data_input, [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)
        
        elif COMBINED_IMAGES == False and WITH_LANDMARKS == True:
            test_input_landmarks = np.load('numpy/X_test_input_landmarks.npy', allow_pickle=True)
            model = custom_vgg_model_w_landmarks(False)
            model.summary()
            opt = Adam(learning_rate = 0.01)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit([inputs_train, inputs_train_landmarks], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_test, inputs_test_landmarks], [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=3, callbacks = [cb_bestModel ,cb_earlyStop])
            model.save_weights("model_checkpoints/model_non_trainable.h5")

            model = custom_vgg_model_w_landmarks(True)
            model.load_weights("model_checkpoints/model_non_trainable.h5")
            model.summary()
            opt = Adam(learning_rate = 0.001)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit([inputs_train, inputs_train_landmarks], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_test, inputs_test_landmarks], [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
            
            model.load_weights("model_checkpoints/model_best.h5")
            result = model.evaluate([test_data_input, test_input_landmarks], [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)
        

        with open('CrossValidation.csv', "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(result)
            wr.writerow(model.metrics_names)
        
        my_dict = scores.history

        if SAVE_PROGRESS_TO_MODEL:
            model.save_weights("model_checkpoints/model_top.h5")
            print("Saved model to disk")

        import pandas as pd
        hist_df = pd.DataFrame(scores.history) 
        with open('numpy/history.json', mode='w') as f:
            hist_df.to_json(f)
        history_df = pd.read_json('numpy/history.json')


    plt.figure(1)
    plt.plot(my_dict['out1_accuracy'])
    plt.plot(my_dict['val_out1_accuracy'])
    plt.plot(my_dict['out1_corr'])
    plt.plot(my_dict['val_out1_corr'])
    plt.plot(my_dict['out1_rmse'])
    plt.plot(my_dict['val_out1_rmse'])
    plt.title('stats for output 1 (valence)')
    plt.ylabel('acc/corr/rmse')
    plt.xlabel('epoch')
    plt.legend(['accuracy: train', 'accuracy: test', 'corr: train', 'corr: test', 'rmse: train', 'rmse: test'], loc='upper left')
    plt.savefig('visualization/output1.png')
    plt.show()

    plt.figure(2)
    plt.plot(my_dict['out2_accuracy'])
    plt.plot(my_dict['val_out2_accuracy'])
    plt.plot(my_dict['out2_corr'])
    plt.plot(my_dict['val_out2_corr'])
    plt.plot(my_dict['out2_rmse'])
    plt.plot(my_dict['val_out2_rmse'])
    plt.title('stats for output 2 (arousal)')
    plt.ylabel('acc/corr/rmse')
    plt.xlabel('epoch')
    plt.legend(['accuracy: train', 'accuracy: test', 'corr: train', 'corr: test', 'rmse: train', 'rmse: test'], loc='upper left')
    plt.savefig('visualization/output2.png')
    plt.show()

    plt.figure(3)
    plt.plot(my_dict['out1_accuracy'])
    plt.plot(my_dict['val_out1_accuracy'])
    plt.title('model accuracy - Valence')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/accuracy_out1.png')
    plt.show()

    plt.figure(4)
    plt.plot(my_dict['out1_corr'])
    plt.plot(my_dict['val_out1_corr'])
    plt.title('model correlation(CORR) - Valence')
    plt.ylabel('correlation')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/correlation_out1.png')
    plt.show()

    plt.figure(5)
    plt.plot(my_dict['out1_rmse'])
    plt.plot(my_dict['val_out1_rmse'])
    plt.title('model root_mean_squared_error - Valence')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/rmse_out1.png')
    plt.show()

    plt.figure(6)
    plt.plot(my_dict['out2_accuracy'])
    plt.plot(my_dict['val_out2_accuracy'])
    plt.title('model accuracy - Arousal')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/accuracy_out2.png')
    plt.show()

    plt.figure(7)
    plt.plot(my_dict['out2_corr'])
    plt.plot(my_dict['val_out2_corr'])
    plt.title('model correlation(CORR) - Arousal')
    plt.ylabel('correlation')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/correlation_out2.png')
    plt.show()

    plt.figure(8)
    plt.plot(my_dict['out2_rmse'])
    plt.plot(my_dict['val_out2_rmse'])
    plt.title('model root_mean_squared_error - Arousal')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/rmse_out2.png')
    plt.show()



def construct_data():
        # SHUFFLE_FOLD = False
        # ORIGINAL_IMAGES = True
        # test_files, test_labels = constructing_data_list_eval(PATH_TO_EVALUATION)
        # test_input = preloading_data(PATH_TO_EVALUATION, test_files)
        # test_target = np.true_divide(test_labels, 10)
        # print("Data to be saved:")
        # print(test_input.shape)
        # print(test_target.shape)
        # np.save('numpy/X_test_input_original.npy', test_input)
        # np.save('numpy/X_test_target_original.npy', test_target)


        global SHUFFLE_FOLD
        SHUFFLE_FOLD = False
        global ORIGINAL_IMAGES
        ORIGINAL_IMAGES = False
        test_files, test_labels = constructing_data_list_eval(PATH_TO_EVALUATION)
        test_input = preloading_data(PATH_TO_EVALUATION, test_files)
        test_target = np.true_divide(test_labels, 10)
        print("Data_2 to be saved:")
        print(test_input.shape)
        print(test_target.shape)
        np.save('numpy/X_test_input.npy', test_input)
        np.save('numpy/Y_test_target.npy', test_target)


        # test_input = np.load('numpy/X_test_input_original.npy')
        # landmarks = []
        # fitter = load_balanced_frontal_face_fitter()
        # detector = MTCNN.detect_faces

        # for img in test_input:
        #     print("New point:")
        #     result = detector.detect(img)
        #     print(result)
        #     point = fitter.fit_from_bb(img, result[1])
        #     print(point)
        #     landmarks.append(point)

        # landmarks = np.array(landmarks)
        # np.save('numpy/landmarks.npy', landmarks)


def construct_facial_landmarks():
    no_face_counter = 0
    total_face_counter = 0
    landmarks_training = []
    landmarks_evaluation = []

    detector = dlib.get_frontal_face_detector() 
    p = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(p) 

    filenames, labels = constructing_data_list(PATH_TO_DATA, FOLD_SIZE)
    for i in FOLD_ARRAY:
        landmarks_train = []
        for name in filenames[i]:
            img= cv2.imread("AFEW-VA/" + str(name)) 
            rect = detector(img)
            print(rect)
            print(len(rect))
            if len(rect) == 0:
                no_face_counter = no_face_counter + 1
                landmarks_train.append(np.zeros((68,)))
                print(no_face_counter)
            else:
                landmarks = predictor(img, rect[0]) 
                landmarks = face_utils.shape_to_np(landmarks)
                landmarks_train.append(landmarks)
                print(landmarks)
            total_face_counter = total_face_counter + 1
        landmarks_training.append(landmarks_train)
    np.save("numpy/landmarks_training.npy", landmarks_training)

    filenames, labels = constructing_data_list_eval(PATH_TO_EVALUATION)
    for name in filenames:
        img = cv2.imread("AFEW-VA_TEST/" + str(name))
        rect = detector(img)
        print(rect)
        print(len(rect))
        if len(rect) == 0:
            no_face_counter = no_face_counter + 1
            print(no_face_counter)
            landmarks_evaluation.append(np.zeros((68, 2)))
        else:
            landmarks = predictor(img, rect[0])
            landmarks = face_utils.shape_to_np(landmarks)
            landmarks_evaluation.append(landmarks)
        total_face_counter = total_face_counter + 1
    np.save("numpy/landmarks_evaluation.npy", landmarks_evaluation)

    print("No-Face-Counter: ")
    print(no_face_counter)
    print("Total-Face-Counter: ")
    print(total_face_counter) 
    
    landmarks_training = np.array(landmarks_training)
    landmarks_evaluation = np.array(landmarks_evaluation)
    print(landmarks_training)
    print(landmarks_evaluation)
    print(landmarks_training.shape)
    print(landmarks_evaluation.shape)



run_model()
print("Training finished")

# construct_data()
# print("Data constructed")

# construct_facial_landmarks()
# print("Landmarks constructed")


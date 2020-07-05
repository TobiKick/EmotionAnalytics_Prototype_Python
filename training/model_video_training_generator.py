#!/usr/bin/env python
# coding: utf-8


############################# IMPORT STATEMENTS ########################################################
#Import Python modules
import numpy as np
from mtcnn import MTCNN
from numpy import asarray
from PIL import Image
import cv2
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.callbacks import ModelCheckpoint, EarlyStopping

#Import Keras modules
from keras.layers import Dense, Flatten, Input, Dropout, Conv1D, Conv2D, LSTM, Concatenate, Reshape, MaxPool1D, MaxPool2D, BatchNormalization
from keras import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.utils import np_utils, Sequence
import keras.backend as K
# from keras.backend import clear_session, set_session

############################# SETUP PROJECT PARAMETERS ########################################################

LOAD_PROGRESS_FROM_MODEL = False
SAVE_PROGRESS_TO_MODEL = True

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

RUN_LOCAL = False
CONSTRUCT_DATA = True

CROSS_VALIDATION = False
NUM_FOLDS = 5

BATCH_SIZE = 32
EPOCHS = 20
EPOCHS_CROSS = 10

PATH_TO_DATA = 'AFEW-VA'
PATH_TO_EXTRACTED_FACES = 'numpy'
DATA_DIR_PREDICT = ''
IMG_FORMAT = '.png'


if RUN_LOCAL == True:
    PATH_TO_DATA = r"C:\Users\Tobias\Desktop\Master-Thesis\Data\AFEW-VA"
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    EPOCHS = 3

# # Approach:
# 
# ### 1. Face detection
# using MTCNN (Simultaneous face detection, face alignment, bounding boxing and landmark detection)
# 
# ### 2. Highlighting faces
# draw the bounding box in an image and plot it - to check out the result
# 
# ### 3. Face extraction
# extracting the face according to the identified bounding box
# 
# ### 4. Face recognition
# Using the VGGFace pretrained Resnet50 model to recognize emotions (training + prediction)

## setting Keras sessions for each network - First Network
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.keras.backend.set_session(sess)
detector = MTCNN()

## Second Network
#sess2 = tf.compat.v1.Session()
#graph = tf.compat.v1.get_default_graph()
#tf.compat.v1.keras.backend.set_session(sess2)
#model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

   
class My_Custom_Generator(Sequence) :
    def __init__(self, image_filenames, labels, batch_size) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
        faces = []
        labels = []
        i = 0
        for file_name in batch_x:
            img = plt.imread(file_name.replace('/', '_'))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces.append(img)
                labels.append(batch_y[i])
            i = i + 1

        return np.array(faces), np.array(labels)


def detect_face(image):
    global sess
    global graph
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        face = detector.detect_faces(image)
        if len(face) == 1:
            return face
        elif len(face) > 1:
            return face[0]
        else:
            print("No face detected")
            return []

def detect_faces(images):
    faces = []  
    global sess
    global graph
    
    face = []
    for img in images:
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)
            face = detector.detect_faces(img)
        if len(face) == 1:
            faces.append(face)  ## just use the face with the highest detection probability
        elif len(face) > 1:
            faces.append(face[0])
        else:
            faces.append([])
            print("No face detected")
    return faces


def extract_face_from_image(image, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    face = detect_face(image) # content of face is a python dict

    if face == []:
        face_image = Image.fromarray(image)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        return face_array
    else:
        # extract the bounding box from the requested face
        box = np.asarray(face[0]['box'])
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


def extract_face_from_images(images, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    faces = detect_faces(images)
    face_images = []
    
    for i in range(len(faces)):
        if faces[i-1] == []: # No face detected
            face_image = Image.fromarray(images[i-1])
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            face_images.append(face_array)
        else:
            # extract the bounding box from the requested face
            if type(faces[i-1]) is list:  # checks whether more than one face was detected
                box = np.asarray(faces[i-1][0]['box'])
                box[box < 0] = 0
                x1, y1, width, height = box
            else:
                box = np.asarray(faces[i-1]['box'])
                box[box < 0] = 0
                x1, y1, width, height = box

            x2, y2 = x1 + width, y1 + height
            # extract the face
            face_boundary = images[i-1][y1:y2, x1:x2]

            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            face_images.append(face_array)
            
    return face_images
    

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


def constructing_data_list(root_data_dir):
    filenames = []
    labels = []
    
    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    f, l = get_labels_from_file(os.path.join(root_data_dir, train_dir, file), train_dir)
                    filenames.extend(f)
                    labels.extend(l)
                    
    return filenames, labels


def preloading_data(filenames):
    for file_name in filenames:
        img = cv2.imread(os.path.join(PATH_TO_DATA, str(file_name)))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = extract_face_from_image(img)
            plt.imsave(file_name.replace('/', '_'), face)
            print("saved")
    return 1


def custom_vgg_model():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    
    for layer in vgg_model.layers: 
        layer.trainable = False
        print(layer.name)
    
    last_layer = vgg_model.get_layer('pool5').output    
    x = Flatten(name='flatten')(last_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(2, activation='tanh')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    
    return custom_vgg_model

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def corr(y_true, y_pred):
    #normalise
    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])  

    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))

    result=top/bottom
    return K.mean(result)

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

def run_model(path_to_data):
    
    if CONSTRUCT_DATA == True:
        #reading in data and appropriately structuring it with a list for the filenames and the labels
        filenames, labels = constructing_data_list(path_to_data)
        # shuffling the data, to have a better learning + randomization
        filenames_shuffled, labels_shuffled = shuffle(filenames, labels)

        filenames_shuffled = np.array(filenames_shuffled)
        labels_shuffled = np.array(labels_shuffled)

        X_train_filenames, X_val_filenames, Y_train_labels, Y_val_labels = train_test_split(
        filenames_shuffled, labels_shuffled, test_size=0.25, random_state=1)

        np.save('numpy/gen_filenames_shuffled.npy', filenames_shuffled)
        np.save('numpy/gen_labels_shuffled.npy', labels_shuffled)
        np.save('numpy/gen_X_train_filenames.npy', X_train_filenames)
        np.save('numpy/gen_X_val_filenames.npy', X_val_filenames)

        Y_train_labels = np.true_divide(Y_train_labels, 5)
        np.save('numpy/gen_Y_train_labels.npy', Y_train_labels)
        Y_val_labels = np.true_divide(Y_val_labels, 5)
        np.save('numpy/gen_Y_val_labels.npy', Y_val_labels)
        print("heyho")
        res_t = preloading_data(X_train_filenames)
        res_v = preloading_data(X_val_filenames)
        print(res_t)
        print(res_v)
    else:
        X_train_filenames = np.load('numpy/gen_X_train_filenames.npy')
        X_val_filenames = np.load('numpy/gen_X_val_filenames.npy')
        
        Y_train_labels = np.load('numpy/gen_Y_train_labels.npy')
        Y_val_labels = np.load('numpy/gen_Y_val_labels.npy')
    
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
    
    mc_best = ModelCheckpoint('model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    mc_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    # K-fold Cross Validation model evaluation
    history_accuracy = []
    history_val_accuracy = []
    history_loss = []
    history_val_loss = []
    history_corr = []
    history_val_corr = []
    history_rmse = []
    history_val_rmse = []


    if CROSS_VALIDATION == True:
        inputs = np.load('numpy/filenames_shuffled.npy')
        targets = np.load('numpy/labels_shuffled.npy')
        for train, test in kfold.split(inputs, targets): 
            ######################################################################
            my_training_batch_generator = My_Custom_Generator(inputs[train], targets[train], BATCH_SIZE)
            my_validation_batch_generator = My_Custom_Generator(inputs[test], targets[test], BATCH_SIZE)
            ######################################################################
        
            embedding_output_shape = 2048
            model = model_top(embedding_output_shape)

            if LOAD_PROGRESS_FROM_MODEL:
                model.load_weights("model_checkpoints/model_top.h5")
                print("Loaded model from disk")

            model.summary()
            model.compile(loss = rmse, optimizer = "adam", metrics = ["accuracy", rmse, corr])

            scores = model.fit_generator(generator=my_training_batch_generator,
                           steps_per_epoch = int(len(train) // BATCH_SIZE),
                           epochs = EPOCHS_CROSS,
                           verbose = 1,
                           validation_data = my_validation_batch_generator,
                           validation_steps = int(len(test) // BATCH_SIZE),
                           callbacks = [mc_best, mc_es])
            
            history_accuracy.extend(scores.history['accuracy'])
            history_val_accuracy.extend(scores.history['val_accuracy'])
            history_loss.extend(scores.history['loss'])
            history_val_loss.extend(scores.history['val_loss'])
            history_corr.extend(scores.history['corr'])
            history_val_corr.extend(scores.history['val_corr'])
            history_rmse.extend(scores.history['rmse'])
            history_val_rmse.extend(scores.history['val_rmse'])
    
    else:
        ######################################################################
        my_training_batch_generator = My_Custom_Generator(X_train_filenames, Y_train_labels, BATCH_SIZE)
        my_validation_batch_generator = My_Custom_Generator(X_val_filenames, Y_val_labels, BATCH_SIZE)
        ######################################################################
        
        model = custom_vgg_model()

        if LOAD_PROGRESS_FROM_MODEL:
            model.load_weights("model_checkpoints/model_top.h5")
            print("Loaded model from disk")

        model.summary()
        model.compile(loss = rmse, optimizer = "adam", metrics = ["accuracy", rmse, corr])
        
        scores = model.fit_generator(generator=my_training_batch_generator,
                           steps_per_epoch = int(len(X_train_filenames) // BATCH_SIZE),
                           epochs = EPOCHS,
                           verbose = 1,
                           validation_data = my_validation_batch_generator,
                           validation_steps = int(len(X_val_filenames) // BATCH_SIZE),
                           callbacks = [mc_best, mc_es])
            
        history_accuracy.extend(scores.history['accuracy'])
        history_val_accuracy.extend(scores.history['val_accuracy'])
        history_corr.extend(scores.history['corr'])
        history_val_corr.extend(scores.history['val_corr'])
        history_rmse.extend(scores.history['rmse'])
        history_val_rmse.extend(scores.history['val_rmse'])
        
    
    if SAVE_PROGRESS_TO_MODEL:
        model.save_weights("model_checkpoints/model_top.h5")
        print("Saved model to disk")

    
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history_accuracy)
    plt.plot(history_val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/accuracy.png')
    plt.show()
    
    # summarize history for CORR
    plt.figure(2)
    plt.plot(history_corr)
    plt.plot(history_val_corr)
    plt.title('model correlation(CORR)')
    plt.ylabel('correlation')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/correlation.png')
    plt.show()
    
    # summarize history for RMSE
    plt.figure(3)
    plt.plot(history_rmse)
    plt.plot(history_val_rmse)
    plt.title('model root_mean_squared_error')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('visualization/rmse.png')
    plt.show()

    
    
run_model(PATH_TO_DATA)
print("Training finished")

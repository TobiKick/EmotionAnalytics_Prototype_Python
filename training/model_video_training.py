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

RUN_LOCAL = False
CONSTRUCT_DATA = False
CROSS_VALIDATION = False
LAYERS_TRAINABLE = False

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

NUM_FOLDS = 5
BATCH_SIZE = 32

PATH_TO_DATA = 'AFEW-VA'
DATA_DIR_PREDICT = ''
IMG_FORMAT = '.png'

if LAYERS_TRAINABLE == True:
    EPOCHS = 1000
    LEARNING_RATE = 0.001
else:
    EPOCHS = 5
    LEARNING_RATE = 0.01

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
sess2 = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.keras.backend.set_session(sess2)
model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


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
    face = detect_face(image)

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


def get_face_embedding(face):
    sample = asarray(face, 'float32')
    # prepare the data for the model
    sample = preprocess_input(sample, version=2)
    
    global sess2
    global graph
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess2)
        output = model_VGGFace.predict(np.array([sample]))[0]
        return output

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
    list_faces = []
    for file_name in filenames:
        img = cv2.imread(os.path.join(PATH_TO_DATA, str(file_name)))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = extract_face_from_image(img)
            list_faces.append(face)

    return np.array(list_faces)


def custom_vgg_model():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    
    for layer in vgg_model.layers: 
        layer.trainable = LAYERS_TRAINABLE
        print(layer.name)
    
    last_layer = vgg_model.get_layer('pool5').output    
    x = Flatten(name='flatten')(last_layer)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
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

        np.save('numpy/filenames_shuffled.npy', filenames_shuffled)
        np.save('numpy/labels_shuffled.npy', labels_shuffled)

        Y_train_labels = np.true_divide(Y_train_labels, 5)
        np.save('numpy/Y_train_labels.npy', Y_train_labels)

        Y_val_labels = np.true_divide(Y_val_labels, 5)
        np.save('numpy/Y_val_labels.npy', Y_val_labels)

        X_train_faces = preloading_data(X_train_filenames)
        np.save('numpy/X_train_faces.npy', X_train_faces)

        X_val_faces = preloading_data(X_val_filenames)
        np.save('numpy/X_val_faces.npy', X_val_faces)

    else:
        Y_train_labels = np.load('numpy/Y_train_labels.npy')
        Y_val_labels = np.load('numpy/Y_val_labels.npy')

        X_train_faces = np.load('numpy/X_train_faces.npy')
        X_val_faces = np.load('numpy/X_val_faces.npy')
    
    np.savetxt("foo.csv", Y_val_labels, delimiter=",")

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
    
    mc_best = ModelCheckpoint('model_checkpoints/model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    mc_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30) # waiting for 10 consecutive epochs that don't reduce the val_loss

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
            embedding_output_shape = 2048
            model = model_top(embedding_output_shape)

            if LOAD_PROGRESS_FROM_MODEL:
                model.load_weights("model_checkpoints/model_top.h5")
                print("Loaded model from disk")

            model.summary()
            opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(loss = rmse, optimizer = opt, metrics = ["accuracy", rmse, corr])

            scores = model.fit(X_train_embeddings, Y_train_labels, validation_data=(X_val_embeddings, Y_val_labels), batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS, callbacks = [mc_best, mc_es])
            
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
        plt.plot(history_accuracy)
        plt.plot(history_val_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/accuracy.png')
        plt.show()

        # summarize history for CORR
        plt.plot(history_corr)
        plt.plot(history_val_corr)
        plt.title('model correlation(CORR)')
        plt.ylabel('correlation')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/correlation.png')
        plt.show()

        # summarize history for RMSE
        plt.plot(history_rmse)
        plt.plot(history_val_rmse)
        plt.title('model root_mean_squared_error')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/rmse.png')
        plt.show()              
    
    else:
        model = custom_vgg_model()

        if LOAD_PROGRESS_FROM_MODEL:
            model.load_weights("model_checkpoints/model_top.h5")
            print("Loaded model from disk")

        model.summary()
        opt = keras.optimizers.Adam(learning_rate = LEARNING_RATE)
        model.compile(loss = rmse, optimizer = opt, metrics = ["accuracy", rmse, corr])
            
        history = model.fit(X_train_faces, Y_train_labels, validation_data=(X_val_faces, Y_val_labels), batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS, callbacks = [mc_best, mc_es])
            
        if SAVE_PROGRESS_TO_MODEL:
            model.save_weights("model_checkpoints/model_top.h5")
            print("Saved model to disk")
        
        
        # summarize history for accuracy
        plt.figure(1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/accuracy.png')
        plt.show()
        
        # summarize history for CORR
        plt.figure(2)
        plt.plot(history.history['corr'])
        plt.plot(history.history['val_corr'])
        plt.title('model correlation(CORR)')
        plt.ylabel('correlation')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/correlation.png')
        plt.show()

        # summarize history for RMSE
        plt.figure(3)
        plt.plot(history.history['rmse'])
        plt.plot(history.history['val_rmse'])
        plt.title('model root_mean_squared_error')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/rmse.png')
        plt.show()

    

   

    
    
run_model(PATH_TO_DATA)
print("Training finished")

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
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, LambdaCallback

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
CONSTRUCT_DATA = True
CROSS_VALIDATION = True
LAYERS_TRAINABLE = False

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

FOLD_NUM = 5
FOLD_ARRAY = [0, 1, 2, 3, 4]
FOLD_SIZE = 120 # number of folders/subjects in one fold

BATCH_SIZE = 32

PATH_TO_DATA = 'AFEW-VA'
PATH_TO_EVALUATION = 'AFEW-VA_EVALUATION'
DATA_DIR_PREDICT = ''
IMG_FORMAT = '.png'

if LAYERS_TRAINABLE == True:
    EPOCHS = 1000
    LEARNING_RATE = 0.0001
else:
    EPOCHS = 3
    LEARNING_RATE = 0.01

if RUN_LOCAL == True:
    PATH_TO_DATA = r"C:\Users\Tobias\Desktop\Master-Thesis\Data\AFEW-VA"
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    EPOCHS = 10
    FOLD_NUM = 2
    FOLD_ARRAY = [0, 1]
    FOLD_SIZE = 1  # number of folders/subjects in one fold

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

    return np.array(filenames), np.array(labels)


def constructing_data_list_crossVal(root_data_dir, fold_size):
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
            filenames_list.append(filenames)
            labels_list.append(labels)
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


def custom_vgg_model():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    
    for layer in vgg_model.layers: 
        layer.trainable = LAYERS_TRAINABLE
        print(layer.name)
    
    last_layer = vgg_model.get_layer('pool5').output    
    x = Reshape((49, 512))(last_layer)
    x = LSTM(64)(x)
    x = Flatten(name='flatten')(last_layer)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    out1 = Dense(1, activation='tanh', name='out1')(x)
    out2 = Dense(1, activation='tanh', name='out2')(x)
    custom_vgg_model = Model(inputs= vgg_model.input, outputs= [out1, out2])
    
    return custom_vgg_model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 


def corr(y_true, y_pred):
    #normalise
    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])  

    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*
                K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))

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

def setTrainable(epoch):
    if epoch == 4:
        this_model = custom_vgg_model()

        for layer in this_model.layers:
            layer.trainable = True

        opt = Adam(learning_rate = 0.01)
        this_model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
        K.set_value(this_model.optimizer.lr, 0.01)

        print("JUhuu, I set the layers to Trainable")

def run_model():
    
    if CONSTRUCT_DATA == True and CROSS_VALIDATION == True:
        filenames, labels = constructing_data_list_crossVal(PATH_TO_DATA, FOLD_SIZE)

        fold_target = np.true_divide(labels, 5)
        np.save('numpy/Y_fold_target.npy', fold_target)

        fold_input = []
        for i in FOLD_ARRAY:
            preload_input = preloading_data(PATH_TO_DATA, filenames[i])
            fold_input.append(preload_input)

        np.save('numpy/X_fold_input.npy', np.array(fold_input))

    elif CONSTRUCT_DATA == True and CROSS_VALIDATION == False:
        #reading in data and appropriately structuring it with a list for the filenames and the labels
        filenames, labels = constructing_data_list(PATH_TO_DATA)
        
        X_train_filenames, X_val_filenames, Y_train_labels, Y_val_labels = train_test_split(
        filenames, labels, test_size=0.25, shuffle=False)  #random_state=1

        Y_train_labels = np.true_divide(Y_train_labels, 5)
        np.save('numpy/Y_train_labels.npy', Y_train_labels)

        Y_val_labels = np.true_divide(Y_val_labels, 5)
        np.save('numpy/Y_val_labels.npy', Y_val_labels)

        X_train_faces = preloading_data(PATH_TO_DATA, X_train_filenames)
        np.save('numpy/X_train_faces.npy', X_train_faces)

        X_val_faces = preloading_data(PATH_TO_DATA, X_val_filenames)
        np.save('numpy/X_val_faces.npy', X_val_faces)

        X_faces = preloading_data(PATH_TO_DATA, filenames)
        np.save('numpy/X_faces.npy', X_faces)

        Y_labels = np.true_divide(labels, 5)
        np.save('numpy/Y_labels', Y_labels)


    if CROSS_VALIDATION == True:
        fold_input = np.load('numpy/X_fold_input.npy')
        fold_target = np.load('numpy/Y_fold_target.npy')
    else:
        Y_train_labels = np.load('numpy/Y_train_labels.npy')
        Y_val_labels = np.load('numpy/Y_val_labels.npy')

        X_train_faces = np.load('numpy/X_train_faces.npy')
        X_val_faces = np.load('numpy/X_val_faces.npy')

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=FOLD_NUM, shuffle=False)
    
    mc_best = ModelCheckpoint('model_checkpoints/model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    mc_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30) # waiting for 10 consecutive epochs that don't reduce the val_loss
    
    mc_crossVal = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    #cb_learningRate =  LearningRateScheduler(scheduler)
    cb_learningRate = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=0.0001)
    cb_layerTrainable = LambdaCallback(
        on_epoch_begin=lambda epoch, logs: setTrainable(epoch))
     
    
    

    # K-fold Cross Validation model evaluation
    history_accuracy_1 = []
    history_val_accuracy_1 = []
    history_corr_1 = []
    history_val_corr_1 = []
    history_rmse_1 = []
    history_val_rmse_1 = []

    history_accuracy_2 = []
    history_val_accuracy_2 = []
    history_corr_2 = []
    history_val_corr_2 = []
    history_rmse_2 = []
    history_val_rmse_2 = []

    if CROSS_VALIDATION == True:
        inputs_train = []
        targets_train = []

        for i in FOLD_ARRAY:
            inputs_test = fold_input[i]
            targets_test = fold_target[i]

            for t in FOLD_ARRAY:
                if t != i:
                    if inputs_train == []:
                        inputs_train = fold_input[t]
                        targets_train = fold_target[t]
                    else:
                        inputs_train = np.concatenate((fold_input[t], inputs_train), axis=0)
                        targets_train = np.concatenate((fold_target[t], targets_train), axis=0)

            model = custom_vgg_model()

            if LOAD_PROGRESS_FROM_MODEL:
                model.load_weights("model_checkpoints/model_top.h5")
                print("Loaded model from disk")

            model.summary()
            opt = Adam(learning_rate = 0.01)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS, callbacks = [mc_crossVal, cb_learningRate, cb_layerTrainable])
            
            result = model.evaluate(inputs_test, [targets_test[:,0], targets_test[:,1]], verbose=1)

            with open('CrossValidation.csv', "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(result)

            history_accuracy_1.extend(scores.history['out1_accuracy'])
            history_corr_1.extend(scores.history['out1_corr'])
            history_rmse_1.extend(scores.history['out1_rmse'])

            history_accuracy_2.extend(scores.history['out2_accuracy'])
            history_corr_2.extend(scores.history['out2_corr'])
            history_rmse_2.extend(scores.history['out2_rmse'])
        
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

        plt.figure(1)
        plt.plot(my_dict['out1_accuracy'])
        plt.plot(my_dict['out1_corr'])
        plt.plot(my_dict['out1_rmse'])
        plt.title('stats for output 1 (valence)')
        plt.ylabel('acc/corr/rmse')
        plt.xlabel('epoch')
        plt.legend(['accuracy: train', 'corr: train', 'rmse: train'], loc='upper left')
        plt.savefig('visualization/output1.png')
        plt.show()

        plt.figure(2)
        plt.plot(my_dict['out2_accuracy'])
        plt.plot(my_dict['out2_corr'])
        plt.plot(my_dict['out2_rmse'])
        plt.title('stats for output 2 (arousal)')
        plt.ylabel('acc/corr/rmse')
        plt.xlabel('epoch')
        plt.legend(['accuracy: train', 'corr: train', 'rmse: train'], loc='upper left')
        plt.savefig('visualization/output2.png')
        plt.show()

        plt.figure(3)
        plt.plot(my_dict['out1_accuracy'])
        plt.title('model accuracy - Valence')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('visualization/accuracy_out1.png')
        plt.show()

        plt.figure(4)
        plt.plot(my_dict['out1_corr'])
        plt.title('model correlation(CORR) - Valence')
        plt.ylabel('correlation')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('visualization/correlation_out1.png')
        plt.show()

        plt.figure(5)
        plt.plot(my_dict['out1_rmse'])
        plt.title('model root_mean_squared_error - Valence')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('visualization/rmse_out1.png')
        plt.show()

        plt.figure(6)
        plt.plot(my_dict['out2_accuracy'])
        plt.title('model accuracy - Arousal')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('visualization/accuracy_out2.png')
        plt.show()

        plt.figure(7)
        plt.plot(my_dict['out2_corr'])
        plt.title('model correlation(CORR) - Arousal')
        plt.ylabel('correlation')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('visualization/correlation_out2.png')
        plt.show()

        plt.figure(8)
        plt.plot(my_dict['out2_rmse'])
        plt.title('model root_mean_squared_error - Arousal')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('visualization/rmse_out2.png')
        plt.show()
    
    else:
        model = custom_vgg_model()

        if LOAD_PROGRESS_FROM_MODEL:
            model.load_weights("model_checkpoints/model_top.h5")
            print("Loaded model from disk")
        
        model.summary()
        opt = Adam(learning_rate = LEARNING_RATE)
        model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
        history = model.fit(X_train_faces, [Y_train_labels[:, 0], Y_train_labels[:,1]], validation_data=(X_val_faces, [Y_val_labels[:,0], Y_val_labels[:,1]]), batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS, callbacks = [mc_best, mc_es])
            
        if SAVE_PROGRESS_TO_MODEL:
            model.save_weights("model_checkpoints/model_top.h5")
            print("Saved model to disk")

        plt.figure(1)
        plt.plot(history.history['out1_accuracy'])
        plt.plot(history.history['val_out1_accuracy'])
        plt.plot(history.history['out1_corr'])
        plt.plot(history.history['val_out1_corr'])
        plt.plot(history.history['out1_rmse'])
        plt.plot(history.history['val_out1_rmse'])
        plt.title('stats for output 1 (valence)')
        plt.ylabel('acc/corr/rmse')
        plt.xlabel('epoch')
        plt.legend(['accuracy: train', 'accuracy: test', 'corr: train', 'corr: test', 'rmse: train', 'rmse: test'], loc='upper left')
        plt.savefig('visualization/output1.png')
        plt.show()

        plt.figure(2)
        plt.plot(history.history['out2_accuracy'])
        plt.plot(history.history['val_out2_accuracy'])
        plt.plot(history.history['out2_corr'])
        plt.plot(history.history['val_out2_corr'])
        plt.plot(history.history['out2_rmse'])
        plt.plot(history.history['val_out2_rmse'])
        plt.title('stats for output 2 (arousal)')
        plt.ylabel('acc/corr/rmse')
        plt.xlabel('epoch')
        plt.legend(['accuracy: train', 'accuracy: test', 'corr: train', 'corr: test', 'rmse: train', 'rmse: test'], loc='upper left')
        plt.savefig('visualization/output2.png')
        plt.show()

        plt.figure(3)
        plt.plot(history.history['out1_accuracy'])
        plt.plot(history.history['val_out1_accuracy'])
        plt.title('model accuracy - Valence')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/accuracy_out1.png')
        plt.show()

        plt.figure(4)
        plt.plot(history.history['out1_corr'])
        plt.plot(history.history['val_out1_corr'])
        plt.title('model correlation(CORR) - Valence')
        plt.ylabel('correlation')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/correlation_out1.png')
        plt.show()

        plt.figure(5)
        plt.plot(history.history['out1_rmse'])
        plt.plot(history.history['val_out1_rmse'])
        plt.title('model root_mean_squared_error - Valence')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/rmse_out1.png')
        plt.show()

        plt.figure(6)
        plt.plot(history.history['out2_accuracy'])
        plt.plot(history.history['val_out2_accuracy'])
        plt.title('model accuracy - Arousal')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/accuracy_out2.png')
        plt.show()

        plt.figure(7)
        plt.plot(history.history['out2_corr'])
        plt.plot(history.history['val_out2_corr'])
        plt.title('model correlation(CORR) - Arousal')
        plt.ylabel('correlation')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/correlation_out2.png')
        plt.show()

        plt.figure(8)
        plt.plot(history.history['out2_rmse'])
        plt.plot(history.history['val_out2_rmse'])
        plt.title('model root_mean_squared_error - Arousal')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('visualization/rmse_out2.png')
        plt.show()

        import pandas as pd
        history_dict = history.history
        hist_df = pd.DataFrame(history.history) 
        
        with open('numpy/history.json', mode='w') as f:
            hist_df.to_json(f)

        history_df = pd.read_json('numpy/history.json')








## EVALUATION ##
def run_evaluation(construct_data):

    if construct_data == True:
        filenames, labels = constructing_data_list(PATH_TO_EVALUATION)
        filenames_shuffled, labels_shuffled = shuffle(filenames, labels)
        X_test_filenames = np.array(filenames_shuffled)
        Y_test_labels = np.array(labels_shuffled)

        X_test_faces = preloading_data(PATH_TO_EVALUATION , X_test_filenames)
        np.save('numpy/X_test_faces.npy', X_test_faces)

        Y_test_labels = np.true_divide(Y_test_labels, 5)
        np.save('numpy/Y_test_labels.npy', Y_test_labels)

    X_test_faces = np.load('numpy/X_test_faces.npy')
    Y_test_labels = np.load('numpy/Y_test_labels.npy')

    print(X_test_faces.shape)
    print(Y_test_labels.shape)

    model = custom_vgg_model()
    opt = Adam(learning_rate = LEARNING_RATE)
    model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})   

    model.load_weights("model_checkpoints/model_best.h5")
    scores = model.evaluate(X_test_faces,  [Y_test_labels[:, 0], Y_test_labels[:,1]], batch_size=BATCH_SIZE, verbose=1)
    
    print(model.metrics_names)
    print(scores)

    with open('evaluation.csv', "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(model.metrics_names)
        wr.writerow(scores)
    

   

    
    
run_model()
print("Training finished")

#run_evaluation(False)
#print("Evaluation finished")
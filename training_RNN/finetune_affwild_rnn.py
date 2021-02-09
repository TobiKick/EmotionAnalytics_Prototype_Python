## modules for data split / sequencing
import os
import json
import numpy as np
from numpy import asarray
from collections import defaultdict
from itertools import chain
import statistics
import csv
from mtcnn import MTCNN
import cv2
from PIL import Image

## modules for NN model
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras.layers import Add, Activation, Dense, Flatten, Input, Dropout, Conv1D, Conv2D, LSTM, Concatenate, Reshape, MaxPool1D, MaxPool2D, BatchNormalization, TimeDistributed, Reshape, GlobalAveragePooling2D
from keras import Model, Sequential
from keras import activations
import keras as keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD 

########################################################################################
########################################################################################

def custom_vgg_model(is_trainable, conv_block, SEQUENCE_LENGTH):
    
    model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    if is_trainable == False:              
        for layer in model_VGGFace.layers:
            layer.trainable = False
    else:
        if conv_block == 1:
            l_name = 'conv5_3_3x3'
        elif conv_block == 2:
            l_name = 'conv5_2_1x1_increase'
        elif conv_block == 3:
            l_name = 'conv5_2_1x1_reduce'
        elif conv_block == 4:
            l_name = 'conv5_1_3x3'

        model_VGGFace.trainable = False
        set_trainable = False
        for layer in model_VGGFace.layers:
            if layer.name == l_name:
                set_trainable = True
            layer.trainable = set_trainable 

    intermediate_model= Model(inputs=model_VGGFace.input, outputs=model_VGGFace.get_layer('avg_pool').output)
    intermediate_model.summary()

    input_tensor = Input(shape=(SEQUENCE_LENGTH, 224, 224, 3))
    x = TimeDistributed( intermediate_model )(input_tensor)

    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(4096, activation='relu', name='dense'))(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = LSTM(128, activation='relu', input_shape=(SEQUENCE_LENGTH, 224, 224, 3), return_sequences=True, name='lstm_1')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    x = LSTM(128, activation='relu', input_shape=(SEQUENCE_LENGTH, 224, 224, 3), return_sequences=True, name='lstm_2')(x)
    # model.add(BatchNormalization(name='batchNorm'))

    out1 = TimeDistributed(Dense(1, activation='tanh'), name='out1')(x)
    out2 = TimeDistributed(Dense(1, activation='tanh'), name='out2')(x)

    model = Model(inputs=[input_tensor], outputs= [out1, out2])
    return model



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
    return K.mean(r)


########################################################################################
########################################################################################

SEQUENCE_LENGTH = 45
BATCH_SIZE = 4

f = np.load('_data_prep/faces_sequenced_45.npy', allow_pickle=True)
v = np.load('_data_prep/valence_sequenced_45.npy', allow_pickle=True)
a = np.load('_data_prep/arousal_sequenced_45.npy', allow_pickle=True)

print(f.shape)
print(v.shape)
print(a.shape)

inputs_train = np.array([])
targets_train_v = np.array([])
targets_train_a = np.array([])
inputs_validation = np.array(f[4])
targets_validation_v = np.array(v[4])
targets_validation_a = np.array(a[4])

for t in [0, 1, 2, 3]:
    if inputs_train.size > 0:
        inputs_train = np.concatenate((np.array(f[t]), inputs_train), axis=0)
        targets_train_v = np.concatenate((np.array(v[t]), targets_train_v), axis=0)
        targets_train_a = np.concatenate((np.array(a[t]), targets_train_a), axis=0)
    else:
        inputs_train = np.array(f[t])
        targets_train_v = np.array(v[t])
        targets_train_a = np.array(a[t])

print(inputs_train.shape)
print(targets_train_v.shape)
print(targets_train_a.shape)
print(inputs_validation.shape)
print(targets_validation_v.shape)
print(targets_validation_a.shape)

cb_bestModel = ModelCheckpoint('model_checkpoints/model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
cb_earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) # waiting for X consecutive epochs that don't reduce the val_loss
# datagen = ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.25,
#     height_shift_range=0.25,
#     horizontal_flip=True,
#     brightness_range=[0.5, 1.5],
#     zoom_range=0.3)

# inputs_train = np.array(inputs_train)
# datagen.fit(inputs_train[1])
# gen1 = datagen.flow(inputs_train, targets_train, batch_size=BATCH_SIZE)
# train_steps = len(gen1)
# train = multi_out(gen1)

model = custom_vgg_model(False, 0, SEQUENCE_LENGTH)
model.summary()
opt = Adam(lr=0.001)  # 0.0001
model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
scores = model.fit(inputs_train, [targets_train_v, targets_train_a], validation_data=(inputs_validation, [targets_validation_v, targets_validation_a]) ,verbose=1, epochs=3)
model.save_weights("model_checkpoints/model_best.h5")

with open('trainHistoryDict_0', 'wb') as file_scores:
    pickle.dump(scores.history, file_scores)

for i in [1, 2, 3, 4]:
    model = custom_vgg_model(True, i, SEQUENCE_LENGTH)
    model.load_weights("model_checkpoints/model_best.h5")
    model.summary()
    opt = Adam(lr=0.0001)
    model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
        
    scores = model.fit(inputs_train, [targets_train_v, targets_train_a], validation_data=(inputs_validation, [targets_validation_v, targets_validation_a]) ,verbose=1, epochs=1000, callbacks = [cb_bestModel ,cb_earlyStop])
    
    with open('trainHistoryDict_' + str(i), 'wb') as file_scores:
        pickle.dump(scores.history, file_scores)



model.load_weights("model_checkpoints/model_best.h5")
result = model.evaluate(inputs_validation, [targets_validation_v, targets_validation_a], verbose=1, batch_size=BATCH_SIZE)

with open('Validation_results.csv', "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(result)
    wr.writerow(model.metrics_names)
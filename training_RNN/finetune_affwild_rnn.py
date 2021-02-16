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
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib
import scikitplot as skplt

## modules for NN model
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras.layers import Add, Activation, Dense, Flatten, Input, Dropout, Conv1D, Conv2D, LSTM, GRU, Concatenate, Reshape, MaxPool1D, MaxPool2D, BatchNormalization, TimeDistributed, Reshape, GlobalAveragePooling2D
from keras import Model, Sequential
from keras import activations
import keras as keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop

########################################################################################
########################################################################################

def custom_vgg_model(SEQUENCE_LENGTH):    
    vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    model_VGGFace= Model(inputs=vggface.input, outputs=vggface.get_layer('avg_pool').output)
    model_VGGFace.summary()

    input_tensor = Input(shape=(SEQUENCE_LENGTH, 224, 224, 3))
    x = TimeDistributed( model_VGGFace )(input_tensor)
    x = TimeDistributed(Flatten())(x)  

    x = TimeDistributed(Dense(1500, activation='relu', name='dense'))(x)
    # x = Dropout(0.2, name='dropout_1')(x)
    # x = GRU(64, activation='relu', return_sequences=True, name='lstm_1')(x)
    # x = Dropout(0.2, name='dropout_2')(x)
    # x = GRU(64, activation='relu', return_sequences=True, name='lstm_2')(x)

    # x = TimeDistributed(Dense(128, activation='relu'), name='dense_2')(x)
    x = TimeDistributed(Dense(512, activation='relu', name='dense'))(x)
    x = TimeDistributed(Dense(128, activation='relu', name='dense'))(x)
    x = TimeDistributed(Dense(32, activation='relu', name='dense'))(x)
    out1 = TimeDistributed(Dense(2, activation='tanh'), name='out1')(x)

    model = Model(inputs=[input_tensor], outputs= [out1])
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


def shape_targets(targets_v, targets_a):
    a = []
    for i in range(0, len(targets_v)):
        b = []
        for j in range(0, 45):
            b.append([targets_v[i][j], targets_a[i][j]])
        a.append(b)

    a = np.array(a)
    print(a.shape)
    return a


def scheduler(epoch, lr):
    if epoch == 0:
        return float(lr)
    else:
        print("New learning rate: " + str(lr *0.95))
        return float(lr * 0.95) 


########################################################################################
########################################################################################

SEQUENCE_LENGTH = 45
BATCH_SIZE = 4   # triggers error with higher batch size because of metrics

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
cb_learningRate =  LearningRateScheduler(scheduler)

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






model = custom_vgg_model(SEQUENCE_LENGTH)

# freezing layers in model_VGGFace
for layer in model.layers[1].layer.layers:
    layer.trainable = False

model.summary()

#opt = RMSprop(learning_rate=0.0001)
opt = Adam(lr=0.0001)
model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr]})  # , 'out2' : ["accuracy", rmse, corr]})

targets_t = shape_targets(targets_train_v, targets_train_a)
targets_val = shape_targets(targets_validation_v, targets_validation_a)
print(targets_t.shape)

scores = model.fit(inputs_train, targets_t, validation_data=(inputs_validation, targets_val) ,verbose=1, epochs=3, batch_size=BATCH_SIZE, callbacks = [cb_bestModel ,cb_earlyStop])
# scores = model.fit(inputs_train, targets_t, validation_data=(inputs_validation, targets_val) ,verbose=1, epochs=3, batch_size=BATCH_SIZE, callbacks = [cb_bestModel ,cb_earlyStop])
# model.save_weights("model_checkpoints/model_best.h5")

with open('trainHistoryDict_0', 'wb') as file_scores:
    pickle.dump(scores.history, file_scores)


for i in [1, 2, 3, 4]:

    model = custom_vgg_model(SEQUENCE_LENGTH)
    try:
        set_trainable = False
        for layer in model.layers[1].layer.layers:
            if i != 1:
                if layer.name == l_name:
                    set_trainable = True
            layer.trainable = set_trainable
        model.load_weights("model_checkpoints/model_best.h5")
    except:
        set_trainable = False
        for layer in model.layers[1].layer.layers:
            if (i-1) != 1:
                if layer.name == l_name_old:
                    set_trainable = True
            layer.trainable = set_trainable
        model.load_weights("model_checkpoints/model_best.h5")

    if i > 1:
        l_name_old = l_name

    if i == 1:
        l_name = 'conv5_3_3x3'
    elif i == 2:
        l_name = 'conv5_2_1x1_increase'
    elif i == 3:
        l_name = 'conv5_2_1x1_reduce'
    elif i == 4:
        l_name = 'conv5_1_3x3'

    set_trainable = False
    for layer in model.layers[1].layer.layers:
        if layer.name == l_name:
            set_trainable = True
        layer.trainable = set_trainable 

    model.summary()

    opt = Adam(lr=0.00001)
    model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr]})     #, 'out2' : ["accuracy", rmse, corr]})
        
    scores = model.fit(inputs_train, targets_t, validation_data=(inputs_validation, targets_val) ,verbose=1, epochs=1000, batch_size=BATCH_SIZE, callbacks = [cb_bestModel ,cb_earlyStop, cb_learningRate])
    
    with open('trainHistoryDict_' + str(i), 'wb') as file_scores:
        pickle.dump(scores.history, file_scores)

    print(targets_val[0:5])
    predict_out = model.predict(inputs_validation[0:5])
    print("Predict output")
    print(predict_out)



model.load_weights("model_checkpoints/model_best.h5")
result = model.evaluate(inputs_validation, [targets_validation_v, targets_validation_a], verbose=1, batch_size=BATCH_SIZE)

with open('Validation_results.csv', "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(result)
    wr.writerow(model.metrics_names)


validation_pred = model.predict(inputs_validation, verbose=1)
np.save('val_labels_v', targets_validation_v)
np.save('val_labels_a', targets_validation_a)
np.save('val_predicted', validation_pred)

skplt.metrics.plot_confusion_matrix(targets_validation_v, validation_pred[:,0], normalize=True)
plt.show()
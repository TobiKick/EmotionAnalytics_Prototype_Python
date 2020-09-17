#!/usr/bin/env python
# coding: utf-8


############################# IMPORT STATEMENTS ########################################################
#Import Python modules
from datetime import datetime
import numpy as np
import cv2
import os
import csv
from imutils import face_utils 

import argparse 
import imutils 
import dlib 

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf

#Import Keras modules
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD 
import keras.backend as K
import keras as keras


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
WITH_HEATMAP = True
LANDMARKS_ONLY = False

REGRESSION = True

FOLD_ARRAY = [0, 1, 2, 3, 4]
FOLD_SIZE = 115 # number of folders/subjects in one fold
BATCH_SIZE = 32

PATH_TO_DATA = 'AFEW-VA'
PATH_TO_EVALUATION = 'AFEW-VA_TEST'

EPOCHS = 1000

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

from _functions import *
from _models import *
   
    

def run_model():
    if CONSTRUCT_DATA == True:
        construct_data(PATH_TO_DATA, PATH_TO_EVALUATION, REGRESSION, ORIGINAL_IMAGES, WITH_LANDMARKS, WITH_HEATMAP, SHUFFLE_FOLD, FOLD_SIZE, FOLD_ARRAY)
    else:
        fold_input_landmarks = np.load('numpy/X_fold_input_landmarks.npy', allow_pickle=True)
        test_input_landmarks = np.load('numpy/X_test_input_landmarks.npy', allow_pickle=True)
        if COMBINED_IMAGES == True:
            fold_input = np.load('numpy/X_fold_input.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target.npy', allow_pickle=True)
            fold_input_2 = np.load('numpy/X_fold_input_original.npy', allow_pickle=True)
            fold_target_2 = np.load('numpy/Y_fold_target_original.npy', allow_pickle=True)

            test_data_input_2 = np.load('numpy/X_test_input_original.npy', allow_pickle=True)
            test_data_input = np.load('numpy/X_test_input.npy', allow_pickle=True)
            test_data_target = np.load('numpy/Y_test_target.npy', allow_pickle=True)

        elif ORIGINAL_IMAGES == True and SHUFFLE_FOLD == True:
            if WITH_HEATMAP == True:
                fold_input = np.load('numpy/X_fold_input_shuffled_original_heatmap.npy', allow_pickle=True)
                test_data_input = np.load('numpy/X_test_input_shuffled_original_heatmap.npy', allow_pickle=True)
            elif WITH_LANDMARKS == True:
                fold_input = np.load('numpy/X_fold_input_shuffled_original_landmarks.npy', allow_pickle=True)
                test_data_input = np.load('numpy/X_test_input_shuffled_original_landmarks.npy', allow_pickle=True)
            else:
                fold_input = np.load('numpy/X_fold_input_shuffled_original.npy', allow_pickle=True)
                test_data_input = np.load('numpy/X_test_input_shuffled_original.npy', allow_pickle=True)
            
            if REGRESSION == True:
                fold_target = np.load('numpy/Y_fold_target_shuffled_original_regr.npy', allow_pickle=True)
                test_data_target = np.load('numpy/Y_test_target_shuffled_original_regr.npy', allow_pickle=True)#
            else: # not updated yet !! first construct data before loading this !!
                fold_target = np.load('numpy/Y_fold_target_shuffled_original.npy', allow_pickle=True)
                test_data_target = np.load('numpy/Y_test_target_shuffled_original.npy', allow_pickle=True)


        elif ORIGINAL_IMAGES == False and SHUFFLE_FOLD == True:
            
            if WITH_HEATMAP == True:
                fold_input = np.load('numpy/X_fold_input_shuffled_heatmap.npy', allow_pickle=True)
                test_data_input = np.load('numpy/X_test_input_shuffled_heatmap.npy', allow_pickle=True)
            elif WITH_LANDMARKS == True:
                fold_input = np.load('numpy/X_fold_input_shuffled_landmarks.npy', allow_pickle=True)
                test_data_input = np.load('numpy/X_test_input_shuffled_landmarks.npy', allow_pickle=True)
            else:
                fold_input = np.load('numpy/X_fold_input_shuffled.npy', allow_pickle=True)
                test_data_input = np.load('numpy/X_test_input_shuffled.npy', allow_pickle=True)
            
            if REGRESSION == True:
                fold_target = np.load('numpy/Y_fold_target_shuffled_regr.npy', allow_pickle=True)
                test_data_target = np.load('numpy/Y_test_target_shuffled_regr.npy', allow_pickle=True)
            else:
                fold_target = np.load('numpy/Y_fold_target_shuffled.npy', allow_pickle=True)
                test_data_target = np.load('numpy/Y_test_target_shuffled.npy', allow_pickle=True)

        elif ORIGINAL_IMAGES == True and SHUFFLE_FOLD == False:
            fold_input = np.load('numpy/X_fold_input_original.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target_original.npy', allow_pickle=True)
            test_data_input = np.load('numpy/X_test_input_original.npy', allow_pickle=True)
            test_data_target = np.load('numpy/Y_test_target_original.npy', allow_pickle=True)      
            
        elif ORIGINAL_IMAGES == False and SHUFFLE_FOLD == False:
            fold_input = np.load('numpy/X_fold_input.npy', allow_pickle=True)
            fold_target = np.load('numpy/Y_fold_target.npy', allow_pickle=True)
            test_data_input = np.load('numpy/X_test_input.npy', allow_pickle=True)
            test_data_target = np.load('numpy/Y_test_target.npy', allow_pickle=True)


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
                opt = Adam(learning_rate = 0.001)
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
                opt = Adam(learning_rate = 0.0001)
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


            if DATA_AUGMENTATION == True and COMBINED_IMAGES == False:
                datagen = ImageDataGenerator(
                    # featurewise_center=True,
                    # featurewise_std_normalization=True,
                    rotation_range=30,
                    width_shift_range=0.25,
                    height_shift_range=0.25,
                    horizontal_flip=True,
                    brightness_range=[0.5, 1.0],
                    zoom_range=0.3)
          
                datagen.fit(inputs_train)
                gen1 = datagen.flow(inputs_train, targets_train, batch_size=BATCH_SIZE)
                train_steps = len(gen1)
                train = multi_out(gen1)

                datagen_val = ImageDataGenerator(
                    # featurewise_center=True,
                    # featurewise_std_normalization=True,
                    rotation_range=30,
                    width_shift_range=0.25,
                    height_shift_range=0.25,
                    horizontal_flip=True,
                    brightness_range=[0.5, 1.0],
                    zoom_range=0.3)

                datagen_val.fit(inputs_test)
                gen2 = datagen_val.flow(inputs_test, targets_test, batch_size=BATCH_SIZE)
                val_steps = len(gen2)
                val = multi_out(gen2)

                model = custom_vgg_model(False, COMBINED_IMAGES, LAYER_REGULARIZATION, REGRESSION)
                model.summary()
                opt = Adam(learning_rate = 0.001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                scores = model.fit(train, steps_per_epoch=train_steps, validation_data=val, validation_steps=val_steps ,verbose=1, epochs=3)
                model.save_weights("model_checkpoints/model_non_trainable.h5")

                model = custom_vgg_model(True, COMBINED_IMAGES, LAYER_REGULARIZATION, REGRESSION)
                model.load_weights("model_checkpoints/model_non_trainable.h5")
                model.summary()
                opt = Adam(learning_rate = 0.0001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                
                scores = model.fit(train, steps_per_epoch=train_steps, validation_data=val, validation_steps=val_steps ,verbose=1, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
                model.load_weights("model_checkpoints/model_best.h5")
                result = model.evaluate(test_data_input, [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)


            # elif DATA_AUGMENTATION == True and COMBINED_IMAGES == False:
            #     datagen = ImageDataGenerator(
            #         # featurewise_center=True,
            #         # featurewise_std_normalization=True,
            #         rotation_range=30,
            #         width_shift_range=0.25,
            #         height_shift_range=0.25,
            #         horizontal_flip=True,
            #         brightness_range=[0.5, 1.0],
            #         zoom_range=0.3)

            #     inputs_train = np.array(inputs_train)

            #     datagen.fit(inputs_train)
            #     gen1 = datagen.flow(inputs_train, targets_train, batch_size=BATCH_SIZE)
            #     train_steps = len(gen1)
            #     train = multi_out(gen1)

            #     model = custom_vgg_model(False, COMBINED_IMAGES, LAYER_REGULARIZATION, REGRESSION)
            #     model.summary()
            #     opt = Adam(learning_rate = 0.001)
            #     model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
            #     scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), validation_steps=(len(inputs_test)/BATCH_SIZE) ,verbose=1, epochs=3)
            #     model.save_weights("model_checkpoints/model_non_trainable.h5")

            #     model = custom_vgg_model(True, COMBINED_IMAGES, LAYER_REGULARIZATION, REGRESSION)
            #     model.load_weights("model_checkpoints/model_non_trainable.h5")
            #     model.summary()
            #     opt = Adam(learning_rate = 0.0001)
            #     model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                
            #     scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), validation_steps=(len(inputs_test)/BATCH_SIZE) ,verbose=1, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
            #     model.load_weights("model_checkpoints/model_best.h5")
            #     result = model.evaluate(test_data_input, [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)


            elif LANDMARKS_ONLY == True:
                model = custom_landmarks_model()
                model.summary()
                opt = Adam(learning_rate = 0.01)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                scores = model.fit(inputs_train_landmarks, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test_landmarks, [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
                
                model.load_weights("model_checkpoints/model_best.h5")
                result = model.evaluate(test_input_landmarks, [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)


            elif COMBINED_IMAGES == True and WITH_LANDMARKS == False:
                model = custom_vgg_model(False)
                model.summary()
                opt = Adam(learning_rate = 0.01)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                scores = model.fit([inputs_train, inputs_train_2], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_test, inputs_test_2], [targets_test[:,0], targets_test[:,1]]), batch_size=BATCH_SIZE, verbose=1, epochs=3)
                model.save_weights("model_checkpoints/model_non_trainable.h5")
                
                model = custom_vgg_model(True)
                model.load_weights("model_checkpoints/model_non_trainable.h5")
                model.summary()
                opt = Adam(learning_rate = 0.001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                scores = model.fit([inputs_train, inputs_train_2], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_test, inputs_test_2], [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
                
                model.load_weights("model_checkpoints/model_best.h5")
                result = model.evaluate([test_data_input, test_data_input_2], [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)

            elif COMBINED_IMAGES == False and WITH_LANDMARKS == False:
                model = custom_vgg_model(False, COMBINED_IMAGES, LAYER_REGULARIZATION)
                model.summary()
                opt = Adam(learning_rate = 0.001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
                model.save_weights("model_checkpoints/model_non_trainable.h5")
                
                model = custom_vgg_model(True, COMBINED_IMAGES, LAYER_REGULARIZATION)
                model.load_weights("model_checkpoints/model_non_trainable.h5")
                model.summary()
                opt = Adam(learning_rate = 0.0001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                
                scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
                model.load_weights("model_checkpoints/model_best.h5")
                result = model.evaluate(test_data_input, [test_data_target[:,0], test_data_target[:,1]], verbose=1, batch_size=BATCH_SIZE)
            
            elif COMBINED_IMAGES == False and WITH_LANDMARKS == True:
                test_input_landmarks = np.load('numpy/X_test_input_landmarks.npy', allow_pickle=True)
                model = custom_vgg_model_w_landmarks(False, COMBINED_IMAGES, LAYER_REGULARIZATION)
                model.summary()
                opt = Adam(learning_rate = 0.001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                scores = model.fit([inputs_train, inputs_train_landmarks], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_test, inputs_test_landmarks], [targets_test[:,0], targets_test[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=3, callbacks = [cb_bestModel ,cb_earlyStop])
                model.save_weights("model_checkpoints/model_non_trainable.h5")

                model = custom_vgg_model_w_landmarks(True, COMBINED_IMAGES, LAYER_REGULARIZATION)
                model.load_weights("model_checkpoints/model_non_trainable.h5")
                model.summary()
                opt = Adam(learning_rate = 0.0001)
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


#################################################################################################
#################################################################################################


run_model()
print("Training finished")
# construct_facial_landmarks()
# print("Landmarks constructed")


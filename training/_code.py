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
import math

import argparse 
import imutils 
import dlib 
import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

#Import Keras modules
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD 
import keras.backend as K
import keras as keras
import pandas as pd
from numpy import asarray


############################# SETUP PROJECT PARAMETERS ########################################################
LOAD_PROGRESS_FROM_MODEL = False
SAVE_PROGRESS_TO_MODEL = True

RUN_LOCAL = False
CONSTRUCT_DATA = True
DATA_GEN = True
CROSS_VALIDATION = True

SHUFFLE_FOLD = True
ORIGINAL_IMAGES = False
LAYER_REGULARIZATION = False
DATA_AUGMENTATION = False

#########################
######## OPTIONS ########
# NOTHING = 0
# WITH_LANDMARKS = 1
# WITH_SOFTATTENTION = 2
# WITH_HEATMAP = 3
# WITH_AAM = 4
# WITH_MASK = 5
#########################
EXTRACT_FACE_OPTION = 5
DISCARD_UNDETECTED_FACES = True  # if with mask == True then also discard undetected faces

#########################
######## OPTIONS ########
# DEFAULT = 0
# WITH LSTM = 1
# WITH MASK = 2
# WITH MASK -> 4 CHANNEL VGGFACE = 3
# WITH MASK -> 4 CHANNEL RESNET 50 = 4
#########################
MODEL_OPTION = 4

FOLD_ARRAY = [0, 1, 2, 3, 4]
FOLD_SIZE = 120
BATCH_SIZE = 16
EPOCHS = 1000

PATH_TO_DATA = './AFEW-VA_ALL'

if RUN_LOCAL == True:
    PATH_TO_DATA = r"C:\Users\Tobias\Desktop\Master-Thesis\Data\AFEW-VA"
    PATH_TO_EVALUATION = r"C:\Users\Tobias\Desktop\Master-Thesis\Data\AFEW-VA_TEST"
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    EPOCHS = 2
    FOLD_ARRAY = [0, 1, 2]   # at least 3 folds necessary to perform split into Train, Val and Test
    FOLD_SIZE = 1  # number of folders/subjects in one fold


########################################################################################################
########################################################################################################
########################################################################################################

from _functions import *
from _models import *

graph_1 = tf.Graph()
graph_2 = tf.Graph()
sess_1 = tf.Session(graph= graph_1)
sess_2 = tf.Session(graph= graph_2)

from PIL import Image

def run_model():
    if CONSTRUCT_DATA == True:
        x, y = constructing_data_list_2(PATH_TO_DATA, FOLD_SIZE)
        #construct_data(PATH_TO_DATA, ORIGINAL_IMAGES, EXTRACT_FACE_OPTION, FOLD_SIZE, FOLD_ARRAY, DISCARD_UNDETECTED_FACES)
    
    elif DATA_GEN == True:
        x = 1
        if x == 0:
            pd_train, pd_val, pd_test = construct_filenames(PATH_TO_DATA, FOLD_SIZE)
            print(pd_train.head())

            t = 0
            df_arr = []
            import pandas as pd
            for elem in [pd_train, pd_val, pd_test]:
                df = pd.DataFrame(columns=["filename", "valence", "arousal"])
                names = elem[['filename']].to_numpy()
               
                print(len(elem[['filename']]))
                for i in range(0, len(elem[['filename']])):
                    img = cv2.imread(os.path.join(PATH_TO_DATA, str(names[i][0])))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        face, mask = extract_mask_from_image(img, ORIGINAL_IMAGES, EXTRACT_FACE_OPTION, DISCARD_UNDETECTED_FACES)
                    
                        if face != []:
                            df = df.append(elem.loc[[i]])
                            plt.imsave("./AFEW-VA_JUST_MASK/" + names[i][0], mask)
                            plt.imsave("./AFEW-VA_JUST_FACE/" + names[i][0], face)
                            #cv2.imwrite("./AFEW-VA_JUST_MASK/" + names[i][0], mask)

                print(df.head())
                if t == 0:
                    df0 = df
                elif t == 1:
                    df1 = df
                else:
                    df2 = df
                t = t + 1

            df0.to_pickle('pd_train.pkl')
            df1.to_pickle('pd_val.pkl')
            df2.to_pickle('pd_test.pkl')

        elif x == 1:
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

            # pd_train, pd_val, pd_test = construct_filenames(PATH_TO_DATA, FOLD_SIZE)
            # print(pd_train.head())
            import pandas as pd
            pd_train = pd.read_pickle('pd_train.pkl')
            print(pd_train.head())
            pd_val = pd.read_pickle('pd_val.pkl')
            pd_test = pd.read_pickle('pd_test.pkl')

            seed = 909 # (IMPORTANT) to transform image and corresponding mask with same augmentation parameter.
            image_datagen = ImageDataGenerator() # width_shift_range=0.1, height_shift_range=0.1, preprocessing_function = image_preprocessing # custom fuction for each image you can use resnet one too.
            mask_datagen = ImageDataGenerator()  # preprocessing_function = mask_preprocessing  # to make mask as feedable formate (256,256,1)
            
            def myGenerator(train_generator_1, train_generator_2):
                while True:
                    xy1 = train_generator_1.next() #or next(train_generator)
                    xy2 = train_generator_2.next() #or next(train_generator1)
                    data = np.concatenate((xy1[0], xy2[0]), axis=3)
                    label = xy1[1]
                    yield (data, label)
            
            def myGenerator_separate(train_generator_1, train_generator_2):
                while True:
                    xy1 = train_generator_1.next() #or next(train_generator)
                    xy2 = train_generator_2.next() #or next(train_generator1)
                    label = xy1[1]
                    yield ([xy1[0], xy2[0]], label)

            image_gen_train = image_datagen.flow_from_dataframe(pd_train, directory="./AFEW-VA_JUST_FACE/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='rgb', seed=seed, shuffle=True)
            mask_gen_train = mask_datagen.flow_from_dataframe(pd_train, directory="./AFEW-VA_JUST_MASK/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='grayscale', seed=seed, shuffle=True)
            train_generator = myGenerator_separate(image_gen_train, mask_gen_train)

            image_gen_val = image_datagen.flow_from_dataframe(pd_val, directory="./AFEW-VA_JUST_FACE/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='rgb', seed=seed, shuffle=True)
            mask_gen_val = mask_datagen.flow_from_dataframe(pd_val, directory="./AFEW-VA_JUST_MASK/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='grayscale', seed=seed, shuffle=True)
            val_generator= myGenerator_separate(image_gen_val, mask_gen_val)

            cb_bestModel = ModelCheckpoint('model_checkpoints/model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            cb_earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) # waiting for X consecutive epochs that don't reduce the val_loss
            # cb_learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
            cb_learningRate =  LearningRateScheduler(scheduler)
            # cb_clr = CyclicLR(base_lr=0.00001, max_lr=0.001, step_size=2000, mode='triangular2')   # 4*(len(inputs_train)/BATCH_SIZE)

            model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
            model.summary()
            
            # opt = Adam(lr = 0.001)
            model.compile(loss = rmse, optimizer ='rmsprop', metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})

            scores = model.fit_generator(train_generator, steps_per_epoch=BATCH_SIZE, validation_data=val_generator, validation_steps=BATCH_SIZE, verbose=1, epochs=1000, callbacks = [cb_bestModel, cb_earlyStop])
            model.save_weights("model_checkpoints/model_top.h5")
            with open('trainHistoryDict_gen', 'wb') as file_scores:
                pickle.dump(scores.history, file_scores)

            history_accuracy_1.extend(scores.history['out1_acc'])
            history_corr_1.extend(scores.history['out1_corr'])
            history_rmse_1.extend(scores.history['out1_rmse'])
            history_accuracy_2.extend(scores.history['out2_acc'])
            history_corr_2.extend(scores.history['out2_corr'])
            history_rmse_2.extend(scores.history['out2_rmse'])

            val_history_accuracy_1.extend(scores.history['val_out1_acc'])
            val_history_corr_1.extend(scores.history['val_out1_corr'])
            val_history_rmse_1.extend(scores.history['val_out1_rmse'])
            val_history_accuracy_2.extend(scores.history['val_out2_acc'])
            val_history_corr_2.extend(scores.history['val_out2_corr'])
            val_history_rmse_2.extend(scores.history['val_out2_rmse'])

            image_gen_eval = image_datagen.flow_from_dataframe(pd_test, directory="./AFEW-VA_ALL/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='rgb', seed=seed, shuffle=True)
            mask_gen_eval = mask_datagen.flow_from_dataframe(pd_test, directory="./AFEW-VA_JUST_MASK/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='grayscale', seed=seed, shuffle=True)
 
            test_generator= myGenerator_separate(image_gen_eval, mask_gen_eval)

            model.load_weights("model_checkpoints/model_best.h5")
            result = model.evaluate_generator(test_generator, steps=BATCH_SIZE, verbose=1)
            
            with open('CrossValidation.csv', "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(result)
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

        else:
            pd_train, pd_val, pd_test = construct_filenames(PATH_TO_DATA, FOLD_SIZE)
            print(pd_train.head())
        
            datagen1 = ImageDataGenerator(data_format='channels_last') #preprocessing_function=myFunc
            gen1 = datagen1.flow_from_dataframe(pd_train, directory="./AFEW-VA_ALL_MASK/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='rgb')
            datagen2 = ImageDataGenerator()
            gen2 = datagen2.flow_from_dataframe(pd_val, directory="./AFEW-VA_ALL_MASK/", x_col="filename", y_col=["valence", "arousal"], class_mode="multi_output", target_size=(224,224), batch_size=BATCH_SIZE, color_mode='rgb')

            cb_bestModel = ModelCheckpoint('model_checkpoints/model_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            cb_earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7) # waiting for X consecutive epochs that don't reduce the val_loss
            # cb_learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
            cb_learningRate =  LearningRateScheduler(scheduler)
            cb_clr = CyclicLR(base_lr=0.00001, max_lr=0.001, step_size=2000, mode='triangular2')   # 4*(len(inputs_train)/BATCH_SIZE)

            model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
            model.summary()
            
            opt = Adam(lr = 0.001)
            model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})

            scores = model.fit_generator(gen1, steps_per_epoch=BATCH_SIZE, validation_data=(gen2), validation_steps=(BATCH_SIZE), verbose=1, epochs=1000, callbacks = [cb_bestModel, cb_earlyStop, cb_learningRate])
            model.save_weights("model_checkpoints/model_top.h5")

    else:
        if ORIGINAL_IMAGES == True:
            if EXTRACT_FACE_OPTION == 0:
                fold_input = np.load('numpy/X_fold_input_original.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 1:
                fold_input = np.load('numpy/X_fold_input_original_landmarks.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 2:
                fold_input = np.load('numpy/X_fold_input_original_softAttention.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 3: # with heatmap
                fold_input = np.load('numpy/X_fold_input_original_heatmap.npy', allow_pickle=True)
            
            fold_target = np.load('numpy/Y_fold_target_original_regr.npy', allow_pickle=True)

        elif ORIGINAL_IMAGES == False:
            if EXTRACT_FACE_OPTION == 0 and DISCARD_UNDETECTED_FACES == True:
                fold_input = np.load('numpy/X_fold_input_discarded.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 0 and DISCARD_UNDETECTED_FACES == False:
                fold_input = np.load('numpy/X_fold_input.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 1:
                fold_input = np.load('numpy/X_fold_input_landmarks.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 2:
                fold_input = np.load('numpy/X_fold_input_softAttention.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 3 and DISCARD_UNDETECTED_FACES == False:
                fold_input = np.load('numpy_old/X_fold_input_heatmap.npy', allow_pickle=True)
                fold_target = np.load('numpy_old/Y_fold_target_heatmap.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 3 and DISCARD_UNDETECTED_FACES == True:
                fold_input = np.load('numpy/X_fold_input_heatmap_discarded.npy', allow_pickle=True)
                fold_target = np.load('numpy/Y_fold_target_heatmap_discarded.npy', allow_pickle=True)
            elif EXTRACT_FACE_OPTION == 5: # with mask
                fold_input = np.load('numpy/X_fold_input_m.npy', allow_pickle=True)
                fold_input_mask = np.load('numpy/X_fold_input_mask.npy', allow_pickle=True)

            # if EXTRACT_FACE_OPTION == 2 and DISCARD_UNDETECTED_FACES == False:
            #     fold_target = np.load('numpy/Y_fold_target_regr_softAttention.npy', allow_pickle=True)
            # elif EXTRACT_FACE_OPTION == 5:
            #     fold_target = np.load('numpy/Y_fold_target_m.npy', allow_pickle=True)
            # elif DISCARD_UNDETECTED_FACES == True:
            #     fold_target = np.load('numpy/Y_fold_target_regr_discarded.npy', allow_pickle=True)
            # else:
            #     fold_target = np.load('numpy/Y_fold_target_regr.npy', allow_pickle=True)

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
                cb_earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) 
                cb_learningRate =  LearningRateScheduler(scheduler)
            
                inputs_test = np.array(fold_input[i])
                targets_test = np.array(fold_target[i])
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
                
                if SHUFFLE_FOLD == True:
                    inputs_train, targets_train = shuffle(inputs_train, targets_train, random_state=0)
                    
                print(inputs_train.shape)
                print(inputs_test.shape)

                model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
                model.summary()
                opt = Adam(lr=0.0001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})

                if DATA_AUGMENTATION == True:
                    datagen = ImageDataGenerator(
                        rotation_range=30,
                        width_shift_range=0.25,
                        height_shift_range=0.25,
                        horizontal_flip=True,
                        brightness_range=[0.5, 1.5],
                        zoom_range=0.3)
            
                    datagen.fit(inputs_train)
                    gen1 = datagen.flow(inputs_train, targets_train, batch_size=BATCH_SIZE)
                    train_steps = len(gen1)
                    train = multi_out(gen1)
                    scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), verbose=1, epochs=3)
                else:
                    scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), verbose=1, batch_size=BATCH_SIZE, epochs=3)
                
                model.save_weights("model_checkpoints/model_best.h5")
                with open('trainHistoryDict_crossVal_' + str(i) + '_0', 'wb') as file_scores:
                    pickle.dump(scores.history, file_scores)

                history_accuracy_1.extend(scores.history['out1_acc'])
                history_corr_1.extend(scores.history['out1_corr'])
                history_rmse_1.extend(scores.history['out1_rmse'])
                history_accuracy_2.extend(scores.history['out2_acc'])
                history_corr_2.extend(scores.history['out2_corr'])
                history_rmse_2.extend(scores.history['out2_rmse'])

                val_history_accuracy_1.extend(scores.history['val_out1_acc'])
                val_history_corr_1.extend(scores.history['val_out1_corr'])
                val_history_rmse_1.extend(scores.history['val_out1_rmse'])
                val_history_accuracy_2.extend(scores.history['val_out2_acc'])
                val_history_corr_2.extend(scores.history['val_out2_corr'])
                val_history_rmse_2.extend(scores.history['val_out2_rmse'])

                for j in [1, 2, 3, 4]:
                    model = custom_vgg_model(True, j, LAYER_REGULARIZATION, MODEL_OPTION)
                    model.load_weights("model_checkpoints/model_best.h5")
                    model.summary()
                    opt = Adam(lr=0.00001)
                    model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                    
                    if DATA_AUGMENTATION:
                        scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), verbose=1, epochs=1000, callbacks = [cb_bestModel, cb_earlyStop])
                    else:
                        scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_test, [targets_test[:,0], targets_test[:,1]]), verbose=1, batch_size=BATCH_SIZE, epochs=1000,  callbacks = [cb_bestModel, cb_earlyStop])

                    model.save_weights("model_checkpoints/model_top.h5")
                    with open('trainHistoryDict_crossVal_' + str(i) + '_1', 'wb') as file_scores:
                        pickle.dump(scores.history, file_scores)

                    history_accuracy_1.extend(scores.history['out1_acc'])
                    history_corr_1.extend(scores.history['out1_corr'])
                    history_rmse_1.extend(scores.history['out1_rmse'])
                    history_accuracy_2.extend(scores.history['out2_acc'])
                    history_corr_2.extend(scores.history['out2_corr'])
                    history_rmse_2.extend(scores.history['out2_rmse'])

                    val_history_accuracy_1.extend(scores.history['val_out1_acc'])
                    val_history_corr_1.extend(scores.history['val_out1_corr'])
                    val_history_rmse_1.extend(scores.history['val_out1_rmse'])
                    val_history_accuracy_2.extend(scores.history['val_out2_acc'])
                    val_history_corr_2.extend(scores.history['val_out2_corr'])
                    val_history_rmse_2.extend(scores.history['val_out2_rmse'])

                model.load_weights("model_checkpoints/model_best.h5")
                result = model.evaluate(inputs_test, [targets_test[:,0], targets_test[:,1]], verbose=1, batch_size=BATCH_SIZE)

                with open('CrossValidation.csv', "a") as fp:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(str(i))
                    wr.writerow(result)
                    
            
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
            cb_earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) # waiting for X consecutive epochs that don't reduce the val_loss
            cb_learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
            cb_clr = CyclicLR(base_lr=0.00001, max_lr=0.001, step_size=2000, mode='triangular2')   # 4*(len(inputs_train)/BATCH_SIZE)
            # cb_learningRate =  LearningRateScheduler(scheduler)
            
            if MODEL_OPTION == 1:
                inputs_fold_LSTM = []
                targets_fold_LSTM = []
                inputs_train_LSTM = []
                targets_train_LSTM = []
                inputs_buffer = []

                for i in FOLD_ARRAY:
                    j = 0
                    for elem in fold_input[i]:
                        inputs_buffer.append(elem) # add new element to fill up list
                        if len(inputs_buffer) == 5:   # always take 5 elements
                            inputs_fold_LSTM.append(np.array(inputs_buffer))
                            targets_fold_LSTM.append(np.array(fold_target[i][j]))
                            inputs_buffer.pop(0)   # pop first element from list   
                        j = j + 1

                    inputs_train_LSTM.append(inputs_fold_LSTM)
                    inputs_fold_LSTM = []
                    targets_train_LSTM.append(targets_fold_LSTM)
                    targets_fold_LSTM = []

                fold_input = np.array(inputs_train_LSTM)
                fold_target = np.array(targets_train_LSTM)

            print(fold_input.shape)
            print(fold_target.shape)

            i = 0
            j = 1
            inputs_test = np.array(fold_input[i])
            targets_test = np.array(fold_target[i])
            inputs_train = []
            targets_train = []

            inputs_validation = np.array(fold_input[j])
            targets_validation = np.array(fold_target[j])

            if MODEL_OPTION >= 2:
                inputs_train_mask = []
                inputs_test_mask = np.array(fold_input_mask[i])
                inputs_validation_mask = np.array(fold_input_mask[j])

            for t in FOLD_ARRAY:
                if t != i and t != j:
                    if inputs_train != []:
                        inputs_train = np.concatenate((fold_input[t], inputs_train), axis=0)
                        targets_train = np.concatenate((fold_target[t], targets_train), axis=0)
                        if MODEL_OPTION >= 2:
                            inputs_train_mask = np.concatenate((fold_input_mask[t], inputs_train_mask), axis=0)
                    else:
                        inputs_train = np.array(fold_input[t])
                        targets_train = np.array(fold_target[t])
                        if MODEL_OPTION >= 2:
                            inputs_train_mask = np.array(fold_input_mask[t])
        
            if SHUFFLE_FOLD == True and MODEL_OPTION >= 2:
                inputs_train, inputs_train_mask, targets_train = shuffle(inputs_train, inputs_train_mask, targets_train, random_state=0)
                inputs_validation, inputs_validation_mask, targets_validation = shuffle(inputs_validation, inputs_validation_mask, targets_validation, random_state=0)
            elif SHUFFLE_FOLD == True:
                inputs_train, targets_train = shuffle(inputs_train, targets_train, random_state=0)
                inputs_validation, targets_validation = shuffle(inputs_validation, targets_validation, random_state=0)
                
            inputs_train = np.array(inputs_train)
            inputs_train = np.squeeze(inputs_train)
            targets_train = np.array(targets_train)
            targets_train = np.squeeze(targets_train)
            inputs_validation = np.squeeze(inputs_validation) 
            targets_validation = np.squeeze(targets_validation)
            inputs_test = np.squeeze(inputs_test)
            targets_test = np.squeeze(targets_test)

            if MODEL_OPTION >= 2:
                inputs_train_mask = np.array(inputs_train_mask)
                inputs_train_mask = np.squeeze(inputs_train_mask)

                # for each channel !!! 
                NORMALIZE = True
                if NORMALIZE == True:
                    inputs_train = normalize_data(inputs_train)
                    inputs_validation = normalize_data(inputs_validation)
                    inputs_test = normalize_data(inputs_test)

            print(inputs_train.shape)
            print(targets_train.shape)

            if DATA_AUGMENTATION == True:
                datagen = ImageDataGenerator(
                    rotation_range=30,
                    width_shift_range=0.25,
                    height_shift_range=0.25,
                    horizontal_flip=True,
                    brightness_range=[0.5, 1.5],
                    zoom_range=0.3)

                inputs_train = np.array(inputs_train)
                datagen.fit(inputs_train)
                gen1 = datagen.flow(inputs_train, targets_train, batch_size=BATCH_SIZE)
                train_steps = len(gen1)
                train = multi_out(gen1)

                model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
                model.summary()
                opt = Adam(lr=0.00005)  # 0.0001
                model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_validation, [targets_validation[:,0], targets_validation[:,1]]), validation_steps=(len(inputs_validation)/BATCH_SIZE) ,verbose=1, epochs=1000, callbacks = [cb_bestModel, cb_earlyStop])
                model.save_weights("model_checkpoints/model_best.h5")

                with open('trainHistoryDict_0', 'wb') as file_scores:
                    pickle.dump(scores.history, file_scores)

                history_accuracy_1.extend(scores.history['out1_acc'])
                history_corr_1.extend(scores.history['out1_corr'])
                history_rmse_1.extend(scores.history['out1_rmse'])
                history_accuracy_2.extend(scores.history['out2_acc'])
                history_corr_2.extend(scores.history['out2_corr'])
                history_rmse_2.extend(scores.history['out2_rmse'])

                val_history_accuracy_1.extend(scores.history['val_out1_acc'])
                val_history_corr_1.extend(scores.history['val_out1_corr'])
                val_history_rmse_1.extend(scores.history['val_out1_rmse'])
                val_history_accuracy_2.extend(scores.history['val_out2_acc'])
                val_history_corr_2.extend(scores.history['val_out2_corr'])
                val_history_rmse_2.extend(scores.history['val_out2_rmse'])

                # for i in [1, 2, 3, 4]:
                #     model = custom_vgg_model(True, i, LAYER_REGULARIZATION, MODEL_OPTION)
                #     model.load_weights("model_checkpoints/model_best.h5")
                #     model.summary()
                #     opt = Adam(lr=0.00001)
                #     model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                     
                #     scores = model.fit(train, steps_per_epoch=train_steps, validation_data=(inputs_validation, [targets_validation[:,0], targets_validation[:,1]]), validation_steps=(len(inputs_validation)/BATCH_SIZE) ,verbose=1, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
                #     with open('trainHistoryDict_' + str(i), 'wb') as file_scores:
                #         pickle.dump(scores.history, file_scores)

                #     history_accuracy_1.extend(scores.history['out1_acc'])
                #     history_corr_1.extend(scores.history['out1_corr'])
                #     history_rmse_1.extend(scores.history['out1_rmse'])
                #     history_accuracy_2.extend(scores.history['out2_acc'])
                #     history_corr_2.extend(scores.history['out2_corr'])
                #     history_rmse_2.extend(scores.history['out2_rmse'])

                #     val_history_accuracy_1.extend(scores.history['val_out1_acc'])
                #     val_history_corr_1.extend(scores.history['val_out1_corr'])
                #     val_history_rmse_1.extend(scores.history['val_out1_rmse'])
                #     val_history_accuracy_2.extend(scores.history['val_out2_acc'])
                #     val_history_corr_2.extend(scores.history['val_out2_corr'])
                #     val_history_rmse_2.extend(scores.history['val_out2_rmse'])

                model.load_weights("model_checkpoints/model_best.h5")
                result = model.evaluate(inputs_test, [targets_test[:,0], targets_test[:,1]], verbose=1, batch_size=BATCH_SIZE)
            
            elif MODEL_OPTION == 1: # with LSTM
                model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
                model.summary()
                opt = Adam(lr = 0.0001)
                model.compile(loss = rmse, optimizer = opt, metrics = {'out' : ["accuracy", rmse, corr]})
                scores = model.fit(inputs_train, targets_train, validation_data=(inputs_validation, targets_validation),verbose=1, batch_size=BATCH_SIZE, epochs=3)
                
                for layer in model.layers:
                    if hasattr(layer, 'layer') and layer.layer.name == 'vggface_resnet50':
                        print(layer.layer.name)
                        layer.layer.save_weights("model_checkpoints/lstm/VGGFace.h5")
                    elif layer.name == 'flatten' or layer.name == 'dropout' or layer.name == 'dropout2':
                        print("Not saved: " + layer.name)
                    else:
                        print(layer.name)
                        weights = layer.get_weights()
                        np.save("model_checkpoints/lstm/weights_" + str(layer.name), weights, allow_pickle=True)

                history_accuracy_1.extend(scores.history['accuracy'])
                history_corr_1.extend(scores.history['corr'])
                history_rmse_1.extend(scores.history['rmse'])
                val_history_accuracy_1.extend(scores.history['val_accuracy'])
                val_history_corr_1.extend(scores.history['val_corr'])
                val_history_rmse_1.extend(scores.history['val_rmse'])
            
                for i in [1, 2, 3, 4]:
                    model = custom_vgg_model(True, i, LAYER_REGULARIZATION, MODEL_OPTION)

                    for layer in model.layers:
                        if hasattr(layer, 'layer') and layer.layer.name == 'vggface_resnet50':
                            print(layer.layer.name)
                            layer.layer.load_weights("model_checkpoints/lstm/VGGFace.h5")
                        elif layer.name == 'flatten' or layer.name == 'dropout' or layer.name == 'dropout2':
                            print("Not set: " + layer.name)
                        else:
                            print(layer.name)
                            weights = np.load("model_checkpoints/lstm/weights_" + str(layer.name) + ".npy", allow_pickle=True)
                            layer.set_weights(weights)

                    model.summary()
                    opt = Adam(lr=0.00001)
                    model.compile(loss = rmse, optimizer = opt, metrics = {'out' : ["accuracy", rmse, corr]})
                    scores = model.fit(inputs_train, targets_train, validation_data=(inputs_validation, targets_validation),verbose=1, batch_size=BATCH_SIZE, epochs=1000, callbacks = [cb_earlyStop])
                
                    for layer in model.layers:
                        if hasattr(layer, 'layer') and layer.layer.name == 'vggface_resnet50':
                            print(layer.layer.name)
                            layer.layer.save_weights("model_checkpoints/lstm/VGGFace.h5")
                        elif layer.name == 'flatten' or layer.name == 'dropout' or layer.name == 'dropout2':
                            print("Not saved: " + layer.name)
                        else:
                            print(layer.name)
                            weights = layer.get_weights()
                            np.save("model_checkpoints/lstm/weights_" + str(layer.name), weights, allow_pickle=True)

                    history_accuracy_1.extend(scores.history['accuracy'])
                    history_corr_1.extend(scores.history['corr'])
                    history_rmse_1.extend(scores.history['rmse'])
                    val_history_accuracy_1.extend(scores.history['val_accuracy'])
                    val_history_corr_1.extend(scores.history['val_corr'])
                    val_history_rmse_1.extend(scores.history['val_rmse'])

                for layer in model.layers:
                    if hasattr(layer, 'layer') and layer.layer.name == 'vggface_resnet50':
                        print(layer.layer.name)
                        layer.layer.load_weights("model_checkpoints/lstm/VGGFace.h5")
                    elif layer.name == 'flatten' or layer.name == 'dropout' or layer.name == 'dropout2':
                        print("Not set: " + layer.name)
                    else:
                        print(layer.name)
                        weights = np.load("model_checkpoints/lstm/weights_" + str(layer.name) + ".npy", allow_pickle=True)
                        layer.set_weights(weights)

                result = model.evaluate(inputs_test, targets_test, verbose=1, batch_size=BATCH_SIZE)

            else:
                
                if MODEL_OPTION == 2:
                    model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
                    model.summary()
                    opt = Adam(lr = 0.0001)
                    model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                
                    scores = model.fit([inputs_train, inputs_train_mask], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_validation, inputs_validation_mask], [targets_validation[:,0], targets_validation[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=3)
                    model.save_weights("model_checkpoints/model_top.h5")
                
                elif MODEL_OPTION == 4: # MASK AS 4TH CHANNEL WITH VGGFACE

                    a = inputs_train_mask
                    inputs_train_mask = np.reshape(a, (a.shape[0], a.shape[1], a.shape[2], 1))
                    inputs_train = np.concatenate((inputs_train, inputs_train_mask), axis=3)
                    print(inputs_train.shape)

                    # datagen = ImageDataGenerator(
                    #     rotation_range=20,
                    #     width_shift_range=0.15,
                    #     height_shift_range=0.15,
                    #     horizontal_flip=True,
                    #     brightness_range=[0.7, 1.3],
                    #     zoom_range=0.2)
                    # datagen.fit(inputs_train)
                    # gen1 = datagen.flow(inputs_train, targets_train, batch_size=BATCH_SIZE)
                    # train_steps = len(gen1)
                    # train = multi_out(gen1)

                    b = inputs_validation_mask
                    inputs_validation_mask = np.reshape(b, (b.shape[0], b.shape[1], b.shape[2], 1))
                    inputs_validation = np.concatenate((inputs_validation, inputs_validation_mask), axis=3)
                    print(inputs_validation.shape)

                    # datagen2 = ImageDataGenerator()
                    # datagen2.fit(inputs_validation)
                    # gen2 = datagen2.flow(inputs_validation, targets_validation, batch_size=BATCH_SIZE)
                    # validation_steps = len(gen2)
                    # validation = multi_out(gen2)
                    
                    model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
                    model.summary()
                    opt = Adam(lr = 0.01)
                    model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                
                    # scores = model.fit_generator(train, steps_per_epoch=train_steps, validation_data=(validation), validation_steps=(validation_steps), verbose=1, epochs=1000, callbacks = [cb_bestModel, cb_earlyStop, cb_learningRate])  #cb_bestModel,
                    scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_validation, [targets_validation[:,0], targets_validation[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=1000, callbacks = [cb_bestModel, cb_earlyStop, cb_learningRate] )  # callbacks = [cb_bestModel, cb_earlyStop]
                    model.save_weights("model_checkpoints/model_last.h5")
                
                else:
                    model = custom_vgg_model(False, 0, LAYER_REGULARIZATION, MODEL_OPTION)
                    model.summary()
                    opt = Adam(lr = 0.0001)
                    model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                
                    scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_validation, [targets_validation[:,0], targets_validation[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=3, callbacks = [cb_bestModel ,cb_earlyStop])
                    model.save_weights("model_checkpoints/model_last.h5")

                history_accuracy_1.extend(scores.history['out1_acc'])
                history_corr_1.extend(scores.history['out1_corr'])
                history_rmse_1.extend(scores.history['out1_rmse'])
                history_accuracy_2.extend(scores.history['out2_acc'])
                history_corr_2.extend(scores.history['out2_corr'])
                history_rmse_2.extend(scores.history['out2_rmse'])

                val_history_accuracy_1.extend(scores.history['val_out1_acc'])
                val_history_corr_1.extend(scores.history['val_out1_corr'])
                val_history_rmse_1.extend(scores.history['val_out1_rmse'])
                val_history_accuracy_2.extend(scores.history['val_out2_acc'])
                val_history_corr_2.extend(scores.history['val_out2_corr'])
                val_history_rmse_2.extend(scores.history['val_out2_rmse'])
            
                if MODEL_OPTION != 4:
                    for i in [1, 2, 3, 4]:
                        model = custom_vgg_model(True, i, LAYER_REGULARIZATION, MODEL_OPTION)
                        model.load_weights("model_checkpoints/model_best.h5")         
                        model.summary()
                        opt = Adam(lr=0.00001)
                        model.compile(loss = rmse, optimizer = opt, metrics = {'out1' : ["accuracy", rmse, corr], 'out2' : ["accuracy", rmse, corr]})
                        
                        if MODEL_OPTION == 2:
                            scores = model.fit([inputs_train, inputs_train_mask], [targets_train[:, 0], targets_train[:,1]], validation_data=([inputs_validation, inputs_validation_mask], [targets_validation[:,0], targets_validation[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [cb_bestModel ,cb_earlyStop])
                        else:
                            scores = model.fit(inputs_train, [targets_train[:, 0], targets_train[:,1]], validation_data=(inputs_validation, [targets_validation[:,0], targets_validation[:,1]]),verbose=1, batch_size=BATCH_SIZE, epochs=1000, callbacks = [cb_bestModel ,cb_earlyStop])  # cb_bestModel
                            model.save_weights("model_checkpoints/model_last.h5")

                history_accuracy_1.extend(scores.history['out1_acc'])
                history_corr_1.extend(scores.history['out1_corr'])
                history_rmse_1.extend(scores.history['out1_rmse'])
                history_accuracy_2.extend(scores.history['out2_acc'])
                history_corr_2.extend(scores.history['out2_corr'])
                history_rmse_2.extend(scores.history['out2_rmse'])

                val_history_accuracy_1.extend(scores.history['val_out1_acc'])
                val_history_corr_1.extend(scores.history['val_out1_corr'])
                val_history_rmse_1.extend(scores.history['val_out1_rmse'])
                val_history_accuracy_2.extend(scores.history['val_out2_acc'])
                val_history_corr_2.extend(scores.history['val_out2_corr'])
                val_history_rmse_2.extend(scores.history['val_out2_rmse'])

                model.load_weights("model_checkpoints/model_best.h5")
                
                if MODEL_OPTION == 4:
                    c = inputs_test_mask
                    inputs_test_mask = np.reshape(c, (c.shape[0], c.shape[1], c.shape[2], 1))
                    inputs_test = np.concatenate((inputs_test, inputs_test_mask), axis=3)
                    print(inputs_test.shape)
                    result = model.evaluate(inputs_test, [targets_test[:,0], targets_test[:,1]], verbose=1, batch_size=BATCH_SIZE)
        
                elif MODEL_OPTION == 2:
                    result = model.evaluate([inputs_test, inputs_test_mask], [targets_test[:,0], targets_test[:,1]], verbose=1, batch_size=BATCH_SIZE)
                
                else:
                    result = model.evaluate(inputs_test, [targets_test[:,0], targets_test[:,1]], verbose=1, batch_size=BATCH_SIZE)


            with open('CrossValidation.csv', "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(result)
                wr.writerow(model.metrics_names)
            
            # my_dict = scores.history
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


            if SAVE_PROGRESS_TO_MODEL:
                model.save_weights("model_checkpoints/model_top.h5")
                print("Saved model to disk")

            import pandas as pd
            hist_df = pd.DataFrame(scores.history) 
            with open('numpy/history.json', mode='w') as f:
                hist_df.to_json(f)
            history_df = pd.read_json('numpy/history.json')

        if MODEL_OPTION == 1:
            plt.figure(1)
            plt.plot(my_dict['out1_accuracy'])
            plt.plot(my_dict['val_out1_accuracy'])
            plt.plot(my_dict['out1_corr'])
            plt.plot(my_dict['val_out1_corr'])
            plt.plot(my_dict['out1_rmse'])
            plt.plot(my_dict['val_out1_rmse'])
            plt.title('stats for output (valence)')
            plt.ylabel('acc/corr/rmse')
            plt.xlabel('epoch')
            plt.legend(['accuracy: train', 'accuracy: test', 'corr: train', 'corr: test', 'rmse: train', 'rmse: test'], loc='upper left')
            plt.savefig('visualization/output.png')
            plt.show()

            plt.figure(3)
            plt.plot(my_dict['out1_accuracy'])
            plt.plot(my_dict['val_out1_accuracy'])
            plt.title('model accuracy - Valence')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('visualization/accuracy_out.png')
            plt.show()

            plt.figure(4)
            plt.plot(my_dict['out1_corr'])
            plt.plot(my_dict['val_out1_corr'])
            plt.title('model correlation(CORR) - Valence')
            plt.ylabel('correlation')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('visualization/correlation_out.png')
            plt.show()

            plt.figure(5)
            plt.plot(my_dict['out1_rmse'])
            plt.plot(my_dict['val_out1_rmse'])
            plt.title('model root_mean_squared_error - Valence')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('visualization/rmse_out.png')
            plt.show()

        else:
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




#################################################################################################
#################################################################################################


run_model()
print("Training finished")
## file: _functions.py
## intended as a support file for _code.py
##

import numpy as np
from mtcnn import MTCNN
from numpy import asarray
from PIL import Image
import cv2
import os
import json

import argparse 
import imutils 
import dlib 

from sklearn.utils import shuffle
import tensorflow as tf
import keras.backend as K
import keras as keras

## from menpofit.io import load_fitter
## from menpofit.aam import load_balanced_frontal_face_fitter

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import imgaug as ia
from imgaug.augmentables.heatmaps import HeatmapsOnImage

###########################################################################
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

IMG_FORMAT = '.png'
###########################################################################


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


def extract_face_from_image(image, original_images, with_landmarks, with_heatmap, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    face_image = Image.fromarray(image)
    face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = asarray(face_image)

    face = detect_face(image)
    print(face)

    if face == []:    
        return image
    else:
        # extract the bounding box from the requested face
        box = np.asarray(face[0].get("box", ""))
        box[box < 0] = 0
        x1, y1, width, height =  box
        x2, y2 = x1 + width, y1 + height

        if with_heatmap == True:
            rect = dlib.rectangle(left=x1, top=y1, right=(x1+width), bottom=(y1+height))
            landmarks = dlib_predictor(image, rect)
            xy  = []
            for n in range(0,68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                xy.append([x,y])
            xy = np.float32(xy)

            kpsoi = ia.KeypointsOnImage.from_xy_array(xy, shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
            distance_maps = kpsoi.to_distance_maps()
            
            max_distance = np.linalg.norm(np.float32([IMAGE_HEIGHT, IMAGE_WIDTH]))
            distance_maps_normalized = distance_maps / max_distance

            heatmaps = HeatmapsOnImage((1.0 - distance_maps_normalized)**100, shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
            img = heatmaps.draw_on_image(image)
            out = np.amax(img, axis=0)   # Maxima along the first axis
            out = np.maximum(out, image) # Maxima between original image and heatmap
            
            if original_images == False:
                face_boundary = out[y1:y2, x1:x2]
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                out = asarray(face_image)
            
            return out

        elif with_landmarks == True:
            rect = dlib.rectangle(left=x1, top=y1, right=(x1+width), bottom=(y1+height))
            landmarks = dlib_predictor(image, rect)
            for n in range(0,68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            out = image

            if original_images == False:
                face_boundary = image[y1:y2, x1:x2]
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                out = asarray(face_image)
            
            return out
        
        else:
            out = image
            
            if original_images == False:
                face_boundary = image[y1:y2, x1:x2]
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                out = asarray(face_image)
            
            return out


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


def constructing_data_list_eval(root_data_dir, with_LSTM):
    filenames = []
    labels = []
    filenames_list = []
    labels_list = []
    
    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    f, l = get_labels_from_file(os.path.join(root_data_dir, train_dir, file), train_dir)
                    if with_LSTM == True:
                        filenames.append(f)
                        labels.append(l)
                    else:
                        filenames.extend(f)
                        labels.extend(l)
        
    return np.array(filenames), np.array(labels)


def constructing_data_list(root_data_dir, fold_size, with_LSTM):
    filenames = []
    labels = []
    filenames_list = []
    labels_list = []
    i = 0

    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    f, l = get_labels_from_file(os.path.join(root_data_dir, train_dir, file), train_dir)
                    if with_LSTM == True:
                        filenames.append(f)
                        labels.append(l)
                    else:
                        filenames.extend(f)
                        labels.extend(l)
        i = i + 1
        print(i)
        if i == fold_size:
            filenames_list.append(np.array(filenames))
            labels_list.append(np.array(labels))
            filenames = []
            labels = []
            i = 0
            print("FOLD DONE")

    return np.array(filenames_list), np.array(labels_list)


def preloading_data(path_to_data, filenames, is_original_images, with_landmarks, with_heatmap):
    list_faces = []
    for file_name in filenames:
        img = cv2.imread(os.path.join(path_to_data, str(file_name)))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = extract_face_from_image(img, is_original_images, with_landmarks, with_heatmap)
            list_faces.append(face)

    return np.array(list_faces)


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
    inverted = label_encoder.inverse_transform([np.savetxtargmax(one_hot_encoded)])
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
    inp = corr(y_true, y_pred)
    out = inp * (-1)
    return out

# def corr_loss(y_true, y_pred):
#     x = y_true
#     y = y_pred
#     mx = K.mean(x)
#     my = K.mean(y)
#     xm, ym = x-mx, y-my
#     r_num = K.sum(tf.multiply(xm,ym))
#     r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
#     r = tf.math.divide_no_nan(r_num, r_den, name="division")
#     #r = r_num / r_den
#     r = K.maximum(K.minimum(r, 1.0), -1.0)
#     return 1 - K.square(r)


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


def construct_data(path_to_data, path_to_evaluation, regression, original_images, with_landmarks, with_heatmap, shuffle_fold, fold_size, fold_array, with_LSTM):
    if with_LSTM == True:
        filenames, labels = constructing_data_list(path_to_data, fold_size, with_LSTM)

        fold_input = []
        fold_target = []

        for i in fold_array:
            fold_i = []
            fold_t = []
            for l in labels[i]:
                if regression == True:
                    target = np.true_divide(l, 10)
                fold_t.append(target)
            
            for f in filenames[i]:
                preload_input = preloading_data(path_to_data, f, original_images, with_landmarks, with_heatmap)
                fold_i.append(preload_input)

            fold_target.append(fold_t)
            fold_input.append(fold_i)

        fold_input = np.array(fold_input)
        fold_target = np.array(fold_target)
        
        test_files, test_labels = constructing_data_list_eval(path_to_evaluation, with_LSTM)
        
        test_input = []
        for subject in test_files:
            out = preloading_data(path_to_evaluation, subject, original_images, with_landmarks, with_heatmap)
            test_input.append(out)

        if with_heatmap == True:
            np.save('numpy/X_fold_input_heatmap_lstm.npy', fold_input)
            np.save('numpy/X_test_input_heatmap_lstm.npy', test_input)
        if with_landmarks == True:
            np.save('numpy/X_fold_input_landmarks_img_lstm.npy', fold_input)
            np.save('numpy/X_test_input_landmarks_img_lstm.npy', test_input)
        else:
            np.save('numpy/X_fold_input_lstm.npy', fold_input)
            np.save('numpy/X_test_input_lstm.npy', test_input)

        if regression == True:
            test_target = []
            for l in test_labels:
                out = np.true_divide(l, 10)
                test_target.append(out)
            test_target = np.array(test_target)
            np.save('numpy/Y_fold_target_regr_lstm.npy', fold_target)
            np.save('numpy/Y_test_target_regr_lstm.npy', test_target)
        else:
            test_target_V = one_hot_encoding(test_labels[:,0])
            test_target_A = one_hot_encoding(test_labels[:,1])
            test_target = np.zeros((len(test_target_V), 2, len(test_target_V[0])))
            for i in range(0, len(test_target_V)):
                test_target[i] = [test_target_V[i], test_target_A[i]]
            np.save('numpy/Y_fold_target_lstm.npy', fold_target)
            np.save('numpy/Y_test_target_lstm.npy', test_target)
            
    elif original_images == True and shuffle_fold == True:
        filenames, labels = constructing_data_list(path_to_data, fold_size, with_LSTM)

        fold_input = []
        fold_target = []
        for i in fold_array:
            f, l = shuffle(filenames[i], labels[i], random_state=0)

            if regression == True:
                target = np.true_divide(l, 10)
            else:
                target_V = one_hot_encoding(l[:,0])
                target_A = one_hot_encoding(l[:,1])
                target = np.zeros((len(target_V), 2, len(target_V[0])))
                for i in range(0, len(target_V)):
                    target[i] = [target_V[i], target_A[i]]

            fold_target.append(target)

            preload_input = preloading_data(path_to_data, f, original_images, with_landmarks, with_heatmap)
            fold_input.append(preload_input)

        fold_target = np.array(fold_target)
        fold_input = np.array(fold_input)

        test_files, test_labels = constructing_data_list_eval(path_to_evaluation, with_LSTM)
        test_input = preloading_data(path_to_evaluation, test_files, original_images, with_landmarks, with_heatmap)    

        if with_landmarks == True:
            if with_heatmap == True:
                np.save('numpy/X_fold_input_shuffled_original_heatmap.npy', fold_input)
                np.save('numpy/X_test_input_shuffled_original_heatmap.npy', test_input)
            else:
                np.save('numpy/X_fold_input_shuffled_original_landmarks.npy', fold_input)
                np.save('numpy/X_test_input_shuffled_original_landmarks.npy', test_input)
        else:
            np.save('numpy/X_fold_input_shuffled_original.npy', fold_input)
            np.save('numpy/X_test_input_shuffled_original.npy', test_input)
        if regression == True:
            test_target = np.true_divide(test_labels, 10)
            np.save('numpy/Y_fold_target_shuffled_original_regr.npy', fold_target)
            np.save('numpy/Y_test_target_shuffled_original_regr.npy', test_target)
        else:
            test_target_V = one_hot_encoding(test_labels[:,0])
            test_target_A = one_hot_encoding(test_labels[:,1])
            test_target = np.zeros((len(test_target_V), 2, len(test_target_V[0])))
            for i in range(0, len(test_target_V)):
                test_target[i] = [test_target_V[i], test_target_A[i]]
            np.save('numpy/Y_fold_target_shuffled_original.npy', fold_target)
            np.save('numpy/Y_test_target_shuffled_original.npy', test_target)
    
        
    elif original_images == False and shuffle_fold == True:
        filenames, labels = constructing_data_list(path_to_data, fold_size, with_LSTM)
        
        fold_input = []
        fold_target = []
        for i in fold_array:
            f, l = shuffle(filenames[i], labels[i], random_state=0)
            
            if regression == True:
                target = np.true_divide(l, 10)
            else:
                target_V = one_hot_encoding(l[:,0])
                target_A = one_hot_encoding(l[:,1])
                target = np.zeros((len(target_V), 2, len(target_V[0])))
                for i in range(0, len(target_V)):
                    target[i] = [target_V[i], target_A[i]]
            
            fold_target.append(target)
            preload_input = preloading_data(path_to_data, f, original_images, with_landmarks, with_heatmap)
            fold_input.append(preload_input)

        fold_input = np.array(fold_input)
        fold_target = np.array(fold_target)        

        test_files, test_labels = constructing_data_list_eval(path_to_evaluation, with_LSTM)
        test_input = preloading_data(path_to_evaluation, test_files, original_images, with_landmarks, with_heatmap)
        
        if with_heatmap == True:
            np.save('numpy/X_fold_input_shuffled_heatmap.npy', fold_input)
            np.save('numpy/X_test_input_shuffled_heatmap.npy', test_input)
        elif with_landmarks == True:
            np.save('numpy/X_fold_input_shuffled_landmarks.npy', fold_input)
            np.save('numpy/X_test_input_shuffled_landmarks.npy', test_input)
        else:
            np.save('numpy/X_fold_input_shuffled.npy', fold_input)
            np.save('numpy/X_test_input_shuffled.npy', test_input)

        if regression == True:
            test_target = np.true_divide(test_labels, 10)
            np.save('numpy/Y_test_target_shuffled_regr.npy', test_target)
            np.save('numpy/Y_fold_target_shuffled_regr.npy', fold_target)
        else:
            test_target_V = one_hot_encoding(test_labels[:,0])
            test_target_A = one_hot_encoding(test_labels[:,1])
            test_target = np.zeros((len(test_target_V), 2, len(test_target_V[0])))
            for i in range(0, len(test_target_V)):
                test_target[i] = [test_target_V[i], test_target_A[i]]
            np.save('numpy/Y_test_target_shuffled.npy', test_target)
            np.save('numpy/Y_fold_target_shuffled.npy', fold_target)


    elif original_images == True and shuffle_fold == False:
        filenames, labels = constructing_data_list(path_to_data, fold_size, with_LSTM)

        fold_input = []
        fold_input_landmarks = []
        fold_target = []
        for i in fold_array:
            target = np.true_divide(labels[i], 10)
            fold_target.append(target)

            preload_input, preload_landmarks = preloading_data(path_to_data, filenames[i], original_images, with_landmarks, with_heatmap)
            fold_input.append(preload_input)
            fold_input_landmarks.append(preload_landmarks)

        fold_target = np.array(fold_target)
        fold_input = np.array(fold_input)
        fold_input_landmarks = np.array(fold_input_landmarks)
        np.save('numpy/Y_fold_target_original.npy', fold_target)
        np.save('numpy/X_fold_input_original.npy', fold_input)
        np.save('numpy/X_fold_input_landmarks.npy', fold_input_landmarks)
            
        test_files, test_labels = constructing_data_list_eval(path_to_evaluation, with_LSTM)
        test_input, test_input_landmarks = preloading_data(path_to_evaluation, test_files, original_images, with_landmarks, with_heatmap)
        test_target = np.true_divide(test_labels, 10)
        np.save('numpy/Y_test_target_original.npy', test_target)
        np.save('numpy/X_test_input_original.npy', test_input)
        np.save('numpy/X_test_input_landmarks.npy', test_input_landmarks)

    elif original_images == False and shuffle_fold == False:
        filenames, labels = constructing_data_list(path_to_data, fold_size, with_LSTM)

        fold_input = []
        fold_target = []
        for i in fold_array:
            l = labels[i]
            if regression == True:
                target = np.true_divide(l, 10)
            else:
                target_V = one_hot_encoding(l[:,0])
                target_A = one_hot_encoding(l[:,1])
                target = np.zeros((len(target_V), 2, len(target_V[0])))
                for i in range(0, len(target_V)):
                    target[i] = [target_V[i], target_A[i]]

            fold_target.append(target)

            preload_input = preloading_data(path_to_data, filenames[i], original_images, with_landmarks, with_heatmap)
            fold_input.append(preload_input)

        fold_input = np.array(fold_input)
        fold_target = np.array(fold_target)
        
        test_files, test_labels = constructing_data_list_eval(path_to_evaluation, with_LSTM)
        test_input = preloading_data(path_to_evaluation, test_files, original_images, with_landmarks, with_heatmap)
        
        if with_heatmap == True:
            np.save('numpy/X_fold_input_heatmap.npy', fold_input)
            np.save('numpy/X_test_input_heatmap.npy', test_input)
        if with_landmarks == True:
            np.save('numpy/X_fold_input_landmarks_img.npy', fold_input)
            np.save('numpy/X_test_input_landmarks_img.npy', test_input)
        else:
            np.save('numpy/X_fold_input.npy', fold_input)
            np.save('numpy/X_test_input.npy', test_input)

        if regression == True:
            test_target = np.true_divide(test_labels, 10)
            np.save('numpy/Y_fold_target_regr.npy', fold_target)
            np.save('numpy/Y_test_target_regr.npy', test_target)
        else:
            test_target_V = one_hot_encoding(test_labels[:,0])
            test_target_A = one_hot_encoding(test_labels[:,1])
            test_target = np.zeros((len(test_target_V), 2, len(test_target_V[0])))
            for i in range(0, len(test_target_V)):
                test_target[i] = [test_target_V[i], test_target_A[i]]
            np.save('numpy/Y_fold_target.npy', fold_target)
            np.save('numpy/Y_test_target.npy', test_target)
            
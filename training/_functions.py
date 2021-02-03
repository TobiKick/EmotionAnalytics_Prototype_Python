## file: _functions.py
## intended as a support file for _code.py
##

import numpy as np
from mtcnn import MTCNN
from numpy import asarray
from numpy import expand_dims
from PIL import Image
import cv2
import os
import json

import argparse 
import imutils 
import dlib 
import pandas as pd

from sklearn.utils import shuffle
import tensorflow as tf
import keras.backend as K
import keras as keras
from keras_vggface.utils import preprocess_input

from menpo.image import Image as menpo_image
from menpo.shape import bounding_box
from menpofit.io import load_fitter
from menpofit.aam import load_balanced_frontal_face_fitter

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import imgaug as ia
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from scipy.ndimage.filters import gaussian_filter
from time import sleep
from collections import defaultdict

###########################################################################
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

IMG_FORMAT = '.png'
###########################################################################


## setting Keras sessions for each network - First Network
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
K.set_session(sess)
detector = MTCNN()

p = "shape_predictor_68_face_landmarks.dat"
dlib_predictor = dlib.shape_predictor(p) 


def detect_face(image):
    global sess
    global graph
    with graph.as_default():
        K.set_session(sess)
        face = detector.detect_faces(image)

    if len(face) >= 1:
        return face
    elif len(face) > 1:
        return face[0]
    else:
        print("No face detected")
        print(face)
        return []



def extract_mask_from_image(image, original_images, extract_face_option, discard_undetected_faces, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    face_image = Image.fromarray(image)
    # face_image = Image.fromarray((image * 255).astype(np.uint8))
    face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = asarray(face_image)

    face = detect_face(image)
    print(face)

    if face == []: 
        return [], []     # discard the image and its label from training
    else:
        # extract the bounding box from the requested face
        box = np.asarray(face[0]["box"])
        print(box)
        box[box < 0] = 0
        x1, y1, width, height =  box
        x2, y2 = x1 + width, y1 + height
            
        rect = dlib.rectangle(left=x1, top=y1, right=(x1+width), bottom=(y1+height))
        landmarks = dlib_predictor(image, rect)

        points = []
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x,y])
        points = np.float32(points)

        pt_map=np.zeros([IMAGE_WIDTH, IMAGE_HEIGHT])
        try:
            for point in points:
                pt_map[int(point[1]),int(point[0])]=1
    
            mask_out = gaussian_filter(pt_map, sigma=1)
        except:
            mask_out=pt_map
            print('Empty Mask')

        out = image  
        if original_images == False:
            mask_boundary = mask_out[y1:y2, x1:x2]
            mask_image = Image.fromarray(mask_boundary)
            mask_image = mask_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            mask_out = asarray(mask_image)

            face_boundary = image[y1:y2, x1:x2]
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            out = asarray(face_image)
        
        return out, mask_out


def extract_face_from_image(image, original_images, extract_face_option, discard_undetected_faces, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    face_image = Image.fromarray(image)
    face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = asarray(face_image)

    face = detect_face(image)
    print(face)

    if face == []: 
        if discard_undetected_faces == True:
            return []      # discard the image and its label from training
        else:
            return image   # the original image
    else:
        # extract the bounding box from the requested face
        box = np.asarray(face[0]["box"])
        print(box)
        box[box < 0] = 0
        x1, y1, width, height =  box
        x2, y2 = x1 + width, y1 + height

        if extract_face_option == 1: # landmarks
            rect = dlib.rectangle(left=x1, top=y1, right=(x1+width), bottom=(y1+height))
            landmarks = dlib_predictor(image, rect)
  
            for n in range(0,68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)  # image, cord (x, y), radius 4, color (0, 0, 255), thickness -1
            
            out = image
            if original_images == False:
                face_boundary = image[y1:y2, x1:x2]
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                out = asarray(face_image)
            return out


        elif extract_face_option == 2: # soft attention
            rect = dlib.rectangle(left=x1, top=y1, right=(x1+width), bottom=(y1+height))
            landmarks = dlib_predictor(image, rect)

            overlay = image.copy()   
            for n in range(0,68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                cv2.circle(overlay, (x, y), 4, (255, 0, 0), -1)  # image, cord (x, y), radius 4, color (0, 0, 255), thickness -1
            alpha = 0.3
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, 0)
            
            out = image
            if original_images == False:
                face_boundary = image[y1:y2, x1:x2]
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                out = asarray(face_image)
            return out
        

        elif extract_face_option == 3: # heatmap
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

        elif extract_face_option == 4: # aam
            aam = load_balanced_frontal_face_fitter()
            bb = bounding_box((x1, y1), (x1+width, y1+height))
            img = menpo_image(image, True)
            result = aam.fit_from_bb(img, bb)
            print(result)
            print(result.final_shape)
            print(result.image)

        else:   # no option
            out = image     
            if original_images == False:
                face_boundary = image[y1:y2, x1:x2]
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                out = asarray(face_image)
            
            return out


# def get_labels_from_file(path_to_file, folder):
#     filenames = []
#     labels = []
   
#     with open(path_to_file) as p:
#         data = json.load(p)
        
#     if not 'frames' in data or len(data['frames']) == 0:     
#         exit(0)
#     else:
#         frames = data['frames']
#         for key, value in frames.items():
#             filenames.append(str(folder + '/' + key + IMG_FORMAT))
#             labels.append([value['valence'], value['arousal']])
    
#     return filenames, labels

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
        
    return np.array(filenames), np.array(labels)

def constructing_data_list_2(root_data_dir, fold_size):
    subjects = defaultdict(dict)
    filenames = []
    filenames_list = []
    labels = []
    labels_list = []

    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    with open(os.path.join(root_data_dir, train_dir, file)) as p:
                        data = json.load(p)
                    frames = data['frames']    
                    actor = data['actor']

                    for key, value in frames.items():
                        try:
                            sub_dict = subjects[actor]
                        except KeyError as e:
                            subjects.update({actor: {}})
                            sub_dict = subjects[actor]
                       
                        try:
                            sub_sub_dict = subjects[actor][train_dir]
                        except KeyError as e:
                            subjects[actor].update({train_dir: {}})
                            sub_sub_dict = subjects[actor][train_dir]

                        sub_sub_dict.update({str(key + IMG_FORMAT): [value['valence'], value['arousal']]})
                        sub_dict.update({str(train_dir): sub_sub_dict})
                        subjects.update({actor: sub_dict})
    
    with open('videoclips.csv', "w") as fp:
        wr = csv.writer(fp)
        count = 0
        for key, value in subjects.items():
            for k,v in value.items():
                count = count + 1
            print(key)
            print(count)
            wr.writerow([str(key) + ": " + str(count)])
            count = 0   

    count = 0
    for key, value in subjects.items():
        for k,v in value.items():
            count = count + 1
    
    fold_size = count / 5  # 5 folds
    i = 0
    

    for key, value in subjects.items():
        for k, v in value.items():
            filenames.append(k)
            labels.append(v)
            i = i + 1

        if i >= fold_size:
            filenames_list.append(np.array(filenames))
            labels_list.append(np.array(labels))
            filenames = []
            labels = []
            i = 0
            print("FOLD DONE")

    filenames_list.append(np.array(filenames))
    labels_list.append(np.array(labels))

    return np.array(filenames_list), np.array(labels_list)



def constructing_data_list(root_data_dir, fold_size):
    subjects = {}
    filenames = []
    filenames_list = []
    labels = []
    labels_list = []

    for train_dir in os.listdir(root_data_dir):
        for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
            for file in files:
                if file[-5:] == '.json':
                    with open(os.path.join(root_data_dir, train_dir, file)) as p:
                        data = json.load(p)
                    frames = data['frames']    
                    actor = data['actor']

                    for key, value in frames.items():
                        try:
                            sub_dict = subjects[actor]
                        except KeyError as e:
                            subjects.update({actor: {}})
                            sub_dict = subjects[actor]
                       
                        sub_dict.update({str(train_dir + '/' + key + IMG_FORMAT): [value['valence'], value['arousal']]})
                        subjects.update({actor: sub_dict})
    
    count = 0
    for key, value in subjects.items():
        for k,v in value.items():
            count = count + 1
    
    fold_size = count / 5  # 5 folds
    i = 0
    

    for key, value in subjects.items():
        for k, v in value.items():
            filenames.append(k)
            labels.append(v)
            i = i + 1

        if i >= fold_size:
            filenames_list.append(np.array(filenames))
            labels_list.append(np.array(labels))
            filenames = []
            labels = []
            i = 0
            print("FOLD DONE")

    filenames_list.append(np.array(filenames))
    labels_list.append(np.array(labels))

    return np.array(filenames_list), np.array(labels_list)

# def constructing_data_list(root_data_dir, fold_size):
#     filenames = []
#     labels = []
#     filenames_list = []
#     labels_list = []
#     i = 0

#     for train_dir in os.listdir(root_data_dir):
#         for subdir, dirs, files in os.walk(os.path.join(root_data_dir, train_dir)):
#             for file in files:
#                 if file[-5:] == '.json':
#                     f, l = get_labels_from_file(os.path.join(root_data_dir, train_dir, file), train_dir)
#                     filenames.extend(f)
#                     labels.extend(l)

#         i = i + 1
#         if i == fold_size:
#             filenames_list.append(np.array(filenames))
#             labels_list.append(np.array(labels))
#             filenames = []
#             labels = []
#             i = 0
#             print("FOLD DONE")

#     return np.array(filenames_list), np.array(labels_list)


def preloading_data(path_to_data, filenames, is_original_images, extract_face_option, discard_undetected_faces):
    list_faces = []
    for file_name in filenames:
        img = cv2.imread(os.path.join(path_to_data, str(file_name)))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face = extract_face_from_image(img, is_original_images, extract_face_option, discard_undetected_faces)

            if face != []:
                face = np.array(face)
                face = face.astype('float32')
                face = expand_dims(face, axis=0)
                face_norm = preprocess_input(face, version=2)

                list_faces.append(face_norm)

    return np.array(list_faces)

def preloading_data_w_labels(path_to_data, filenames, labels, is_original_images, extract_face_option, discard_undetected_faces):
    list_faces = []
    list_masks = []
    list_labels = []

    for i in range(0, len(filenames)):
        print(i)
        img = cv2.imread(os.path.join(path_to_data, str(filenames[i])))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if extract_face_option == 5: # with mask
                face, mask = extract_mask_from_image(img, is_original_images, extract_face_option, discard_undetected_faces)
                # face = np.array(face)
                # face = face.astype('float32')
                # face = expand_dims(face, axis=0)
                # face = preprocess_input(face, version=2)
            else:
                face = extract_face_from_image(img, is_original_images, extract_face_option, discard_undetected_faces)

            if face != []:
                list_faces.append(face)
                list_labels.append(labels[i-1])
                if extract_face_option == 5: # with mask
                    list_masks.append(mask)

        else:
            print("Image not found!!")
    
    if extract_face_option == 5:
        return np.array(list_faces), np.array(list_labels), np.array(list_masks)
    else:
        return np.array(list_faces), np.array(list_labels)


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

# import tensorflow_probability as tfp

# def corr(y_true, y_pred):
#     x = tfp.stats.correlation(y_true, y_pred)
#     return x

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

def corr_loss(y_true, y_pred):
    inp = corr(y_true, y_pred)
    out = inp * (-1)
    return out

def total_loss(y_true, y_pred):
    loss = corr_loss(y_true, y_pred) + rmse(y_true, y_pred)
    return loss


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
    
    if epoch == 0:
        print("New learning rate: " + str(lr))
        return float(lr)
    else:
        print("New learning rate: " + str(lr *0.9))
        return float(lr * 0.9) 


def multi_out(gen):
    for x, y in gen:
        yield x, [y[:,0], y[:,1]]

def construct_filenames(path_to_data, fold_size):
    filenames, labels = constructing_data_list(path_to_data, fold_size)

    fn_test = filenames[0]
    labels_test = labels[0]

    fn_val = filenames[1]
    labels_val = labels[1]

    fn_train = np.concatenate((filenames[2], filenames[3], filenames[4]), axis=0)
    labels_train = np.concatenate((labels[2], labels[3], labels[4]), axis=0)

    
    train = np.column_stack((fn_train, labels_train))
    df1 = pd.DataFrame(data=train, columns=["filename", "valence", "arousal"])

    val = np.column_stack((fn_val, labels_val))
    df2 = pd.DataFrame(data=val, columns=["filename", "valence", "arousal"])

    test = np.column_stack((fn_test, labels_test))
    df3 = pd.DataFrame(data=test, columns=["filename", "valence", "arousal"])

    return df1, df2, df3

def normalize_data_4_channels(arr):
    arr_shape = arr.shape
    print(arr_shape)
    arr = arr.reshape(-1, 4)

    scaler = MinMaxScaler()  # default range [0,1]
    scaler.fit(arr)
    arr = scaler.transform(arr)
    arr = arr.reshape(arr_shape)
    return arr


def normalize_data(arr):
    # x_temp = x_temp[..., ::-1]

    arr_shape = arr.shape
    print(arr_shape)
    arr = arr.reshape(-1, 3)

    scaler = MinMaxScaler()  # default range [0,1]
    scaler.fit(arr)
    arr = scaler.transform(arr)
    arr = arr.reshape(arr_shape)
    # scaler.fit(x_temp[..., 0])
    # x_temp[..., 0] = scaler.transform(x_temp[..., 0])
    # scaler.fit(x_temp[..., 1])
    # x_temp[..., 1] = scaler.transform(x_temp[..., 1])
    # scaler.fit(x_temp[..., 2])
    # x_temp[..., 2] = scaler.transform(x_temp[..., 2])

    return arr

def construct_data(path_to_data, original_images, extract_face_option, fold_size, fold_array, discard_undetected_faces):
    if original_images == True:
        filenames, labels = constructing_data_list(path_to_data, fold_size)
        
        fold_input = []
        fold_target = []
        for i in fold_array:
            preload_input, preload_target = preloading_data_w_labels(path_to_data, filenames[i], labels[i], original_images, extract_face_option, discard_undetected_faces)
            fold_input.append(preload_input)
            
            preload_target = np.true_divide(preload_target, 10)
            fold_target.append(preload_target)         

        fold_input = np.array(fold_input)
        fold_target = np.array(fold_target)  

        if extract_face_option == 1: # landmarks
            np.save('numpy/X_fold_input_original_landmarks.npy', fold_input)
        elif extract_face_option == 2: # soft attention
            np.save('numpy/X_fold_input_original_softAttention.npy', fold_input)
        elif extract_face_option == 3: # heatmap
            np.save('numpy/X_fold_input_original_heatmap.npy', fold_input)
        elif extract_face_option == 0:
            np.save('numpy/X_fold_input_original.npy', fold_input)

        np.save('numpy/Y_fold_target_original_regr.npy', fold_target)
        
        
    elif original_images == False:
        filenames, labels = constructing_data_list(path_to_data, fold_size)

        fold_input = []
        fold_target = []
        fold_input_mask = []
        for i in fold_array:
            if extract_face_option == 5:
                preload_input, preload_target, preload_mask = preloading_data_w_labels(path_to_data, filenames[i], labels[i], original_images, extract_face_option, discard_undetected_faces)               
                fold_input.append(preload_input)
                fold_input_mask.append(preload_mask)
            else:
                preload_input, preload_target = preloading_data_w_labels(path_to_data, filenames[i], labels[i], original_images, extract_face_option, discard_undetected_faces)
                fold_input.append(preload_input)
            
            preload_target = np.true_divide(preload_target, 10)
            fold_target.append(preload_target)         

        fold_input = np.array(fold_input)
        fold_target = np.array(fold_target)  

        if extract_face_option == 5:
            faces = fold_input.astype('float32')
            faces = expand_dims(faces, axis=0)
            fold_input = preprocess_input(faces, version=2)
            fold_input_mask = np.array(fold_input_mask)

        if extract_face_option == 1 and discard_undetected_faces == False: # landmarks
            np.save('numpy/X_fold_input_landmarks.npy', fold_input)
        elif extract_face_option == 2 and discard_undetected_faces == False: # soft attention
            np.save('numpy/X_fold_input_softAttention.npy', fold_input)
        elif extract_face_option == 3 and discard_undetected_faces == False: # heatmap
            np.save('numpy/X_fold_input_heatmap.npy', fold_input)
            np.save('numpy/Y_fold_target_heatmap.npy', fold_target)
        elif extract_face_option == 3 and discard_undetected_faces == True: # heatmap
            np.save('numpy/X_fold_input_heatmap_discarded.npy', fold_input)
            np.save('numpy/Y_fold_target_heatmap_discarded.npy', fold_target)
        elif extract_face_option == 5: # mask
            np.save('numpy/X_fold_input_mask.npy', fold_input_mask)
            np.save('numpy/X_fold_input_m.npy', fold_input)
            np.save('numpy/Y_fold_target_m.npy', fold_target)
        elif extract_face_option == 0: # no setting
            np.save('numpy/X_fold_input.npy', fold_input)

        # if discard_undetected_faces == False:
        #     np.save('numpy/Y_fold_target_regr.npy', fold_target)
        # elif discard_undetected_faces == True and extract_face_option != 5:
        #     np.save('numpy/Y_fold_target_regr_discarded.npy', fold_target)



################################################################################################

from keras.callbacks import *
from keras import backend as K
import numpy as np

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
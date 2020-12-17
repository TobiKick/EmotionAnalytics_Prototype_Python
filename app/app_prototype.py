#!/usr/bin/python

# import statements
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSignal
import time
from datetime import datetime
import pygetwindow as gw

# for reading from an window
import cv2
from PIL import ImageGrab, Image
from numpy import asarray
import numpy as np

#For local CPU usage:
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
# from mtcnn.mtcnn import MTCNN
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras.backend import clear_session, set_session
import tensorflow as tf

# VARIABLES
IPYTHON = True
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

INTERVAL_SECONDS = 3  ## Interval seconds between each time an emotion recognition output is produced
IMAGES_PER_INTERVAL = 6
INTEREST_FRAMES = 20

LAYERS_TRAINABLE = True


def getWindowNames():
    visibleWindows = []   
    
    #if gw.getWindowsWithTitle('Fotocamera'):
    #    visibleWindows.append('Fotocamera')
    
    for i in gw.getAllTitles():        
        notepadWindow = gw.getWindowsWithTitle(i)[0]
        
        if notepadWindow.isMinimized or notepadWindow.isMaximized:
            print(notepadWindow.title)
            visibleWindows.append(notepadWindow.title)
    
    return visibleWindows


print(getWindowNames())


def custom_vgg_model():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    
    for layer in vgg_model.layers: 
        layer.trainable = LAYERS_TRAINABLE
        print(layer.name)
    
    last_layer = vgg_model.get_layer('pool5').output    
    x = Flatten(name='flatten')(last_layer)
    # x = Dense(256, activation='relu')(x)
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

model_top = custom_vgg_model()


## setting Keras sessions for each of the pretrained networks
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
detector = MTCNN()

## Second Network
# sess2 = tf.Session()
# graph = tf.get_default_graph()
# set_session(sess2)
# model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Third Network
sess3 = tf.Session()
graph = tf.get_default_graph()
set_session(sess3)
model_top.load_weights("model_best.h5")


def detect_faces(image):
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        faces = detector.detect_faces(image)
        return np.array(faces)
        
        
def extract_face_from_image(image, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    face = detect_faces(image) # content of face is a python dict

    if len(face) == 0:
        return []
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
        

THRESHOLD = 0.5  # range of values is from -1 to +1
emotion_scores_list = []

def identify_interest(val):
    positive = 0
    negative = 0
    neutral = 0

    for i in emotion_scores_list:
        if i > (THRESHOLD/2):
            positive = positive + 1
        elif i < (-THRESHOLD/2):
            negative = negative + 1
        else:
            neutral = neutral + 1

    print(positive)
    print(negative)
    print(neutral)

    if (positive-negative) > 0:
        print((positive-negative)/positive)
    else:
        print((negative-positive)/negative)
    
    if neutral > (positive + negative):
        return "-"
    elif positive/2 > negative:
        return "VERY INTERESTED"
    elif positive > negative:
        return "INTERESTED"
    else:
        return "Not interested!"
        
        
def return_prediction(path):
        #converting image to RGB color and save it
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        #detect face in image, crop it then resize it
        face = extract_face_from_image(img)
        
        if face == []:
            return None, None
        else:
            print("Face detected")
            if face.ndim == 3:
                face = face.reshape((1, face.shape[0], face.shape[1], face.shape[2]))
            face = np.array(face)
            
            clear_session()
            global sess3
            global graph
            with graph.as_default():
                set_session(sess3)
                out1, out2 = model_top.predict(face) #make prediction and display the result
                val = out1[0][0] * 10
                ar = out2[0][0] * 10
                print("result: " + str(val) + ", " + str(ar))
                return val, ar    
                
                
class ProcessStream(QThread):   
    output = pyqtSignal(object)
    def __init__(self):
        super(ProcessStream, self).__init__()
        self.active = True

    def run(self):
        # Task 1: Use Multithreading ???   for GUI + Input Stream
        # Task 2: Get pixels as an Input Stream (using OpenCV)
        # Task 3: Preprocessing & Landmark detection
        # Task 4: Read in Python Model
        image = ImageGrab.grab()
        height,width,channel = np.array(image).shape

        out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, (width,height))
        
        val_list = []
        arr_list = []
        i = 0
        
        while self.active == True:
            # image = ImageGrab.grab(rect)
            image = ImageGrab.grab()
            out.write(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
            
            imgUMat = np.float32(image)
            cv2.imwrite("test.jpg", imgUMat)

            val, arr = return_prediction("test.jpg")
            if val != None:
                val_list.append(val)
                arr_list.append(arr)
                if len(val_list) > IMAGES_PER_INTERVAL:
                    val_list.pop(0)
                    arr_list.pop(0)
                
                emotion_scores_list.append(val)
                if len(emotion_scores_list) > INTEREST_FRAMES:
                    emotion_scores_list.pop(0)
                
                i = i + 1
                if i % INTERVAL_SECONDS == 0:
                    interest = identify_interest(val)
                    print("Interest: " + str(interest))
                
                    valence = sum(val_list) / len(val_list)
                    arousal = sum(arr_list) / len(arr_list)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    result = str(now) + "   Valence: " + str(round(valence, 4)) + "\n" + str(now) + "   Arousal: " + str(round(arousal, 4)) + "\n" + "Conclusion: " + str(interest)
                    self.output.emit(result)
                
            time.sleep(int(INTERVAL_SECONDS/IMAGES_PER_INTERVAL))
        out.release()
        
    def stop(self):
        print("STOP Thread")
        self.active = False
        
        
# Create GUI with PyQt
selected_window = ""

class Ui_MainWindow(object): 
    def setupUi(self, window): 
        super().__init__()
        self.window = window
        #self.window.box_selection.addItems(inputList)
        self.window.btn_start.clicked.connect(self.getSelection)
        self.recording = False
        self.totalResults = ""
        #self.thread1 = ProcessStream(self.window.box_selection, window)
        self.thread1 = ProcessStream()
        self.thread1.output.connect(self.addResults)
        app.aboutToQuit.connect(self.closeEvent)
        
    def getSelection(self): 
        if self.recording == False:
            #selected_window = str(self.window.box_selection.currentText())
            #print("Analytics got started! Selected window: " + selected_window)
            print("Analytics got started!")
            self.recording = True
            # changing the text of label after button got clicked 
            self.window.btn_start.setText("Stop Analytics")
            self.thread1.start() # This actually causes the thread to run
        else:
            self.recording = False
            self.thread1.stop()
                        
            self.window.btn_start.setText("Start Recording")
            # self.thread1 = ProcessStream(self.window.box_selection, self.window)  # recreate thread
            self.thread1 = ProcessStream()
            self.thread1.output.connect(self.addResults)
    
    def closeEvent(self):
        print('Close button pressed')
        self.recording = False
        self.thread1.stop()
        
        if IPYTHON:
            app.deleteLater
        else:
            sys.exit(0)
    
    def addResults(self, inputText):
        self.totalResults = (inputText + "\n" + self.totalResults)
        self.window.box_results.setText(self.totalResults)


if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)  
    window = uic.loadUi("dialog_2.ui")
    
    myApp = Ui_MainWindow()  
    # myApp.setupUi(window, getWindowNames())
    myApp.setupUi(window)  
      
      
    if IPYTHON == False:
        window.show() 
        sys.exit(app.exec_())
    else:
        window.show()
        app.exec_()
        
        

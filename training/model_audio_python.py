# Project Master Thesis - Machine Learning (ML): Emotion Recognition"

############################# IMPORT STATEMENTS ########################################################
#Import Python modules
import librosa as librosa
import numpy as np
import scipy as sp
import scipy.io.wavfile as wav
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt

#Import Keras modules
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Dense, Flatten, Input, Conv2D, LSTM, Concatenate, Reshape, MaxPool2D, BatchNormalization
from keras import Model
from keras.optimizers import Adam
import keras.backend as K

############################# SETUP PROJECT PARAMETERS ########################################################
LOAD_PROGRESS_FROM_MODEL = False
SAVE_PROGRESS_TO_MODEL = False

############################# SETUP AGENT ############ ########################################################

class Singleton:
    # SINGLETON class - Design Pattern
    __instance = None

    def __init__(self, wav_file_location, wav_filename, nr_emotions, state_size_x, state_size_y):
        if Singleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Singleton.__instance = self
            self.nr_emotions = nr_emotions
            self.wav_file_location = wav_file_location
            self.wav_filename = wav_filename
            self.state_size_x = state_size_x
            self.state_size_y = state_size_y
            self.learning_rate = 0.001
            self.totalEpisodes = 10
            #self.model = self._build_model()

    def _build_model(self):
        input = Input(shape=(self.state_size_x, self.state_size_y, 4))

        model = Conv2D(32, kernel_size=(8,8), strides=4, activation='elu')(input)
        model = BatchNormalization()(model)
        model = MaxPool2D((2,2), padding='same')(model)
        model = Conv2D(64, kernel_size=(4,4), strides=2, activation='elu')(model)
        model = BatchNormalization()(model)
        model = MaxPool2D((2,2), padding='same')(model)
        model = Conv2D(64, kernel_size=(3,3), strides=1, activation='elu')(model)
        model = BatchNormalization()(model)
        model = MaxPool2D((2,2), padding='same')(model)
        model = Conv2D(64, kernel_size=(3,3), strides=1, activation='elu')(model)
        model = BatchNormalization()(model)
        model = MaxPool2D((2,2), padding='same')(model)
        model = Flatten()(model)

        model = LSTM(64, input_shape=(657, 1))(model)
        output = Dense(32, activation='relu')(model)

        out = Dense(self.nr_emotions, activation='softmax')(output)

        model = Model(inputs=input, outputs=out)
        model.compile(loss = 'mse',  loss_weights = 0.5, optimizer=Adam(lr=self.learning_rate))

        model.summary()
        return model


    def display_Data(self, data):
        ipd.Audio(self.wav_file_location + self.wav_filename)

        #plt.figure(figsize=(12, 4))
        #librosa.display.waveplot(data, sr=sampling_rate)


    def data_prep(self):
        # link = self.wav_file_location + self.wav_filename
        # data = wav.read(link)
        return data

    def train(self, states, agent_info, actions, actions_oneHot, rewards):
        discounted_rewards = 0
        state_values = self.model.predict([states, agent_info, actions])
        state_values = state_values[1]
        advantages = discounted_rewards - np.reshape(state_values[0], len(state_values[0]))
        weights = {'o_Policy': advantages, 'o_Value': np.ones(len(advantages))}
        self.model.fit([states, agent_info, actions], [actions, discounted_rewards], epochs=1, sample_weight=weights, verbose=0)

    def saveModel(self):
        self.model.save_weights("app_model/model.h5")
        print("TotalEpisodes: " + str(self.totalEpisodes))
        print("Saved model to disk")

    def loadModel(self):
        self.model.load_weights("app_model/model.h5")
        print("TotalEpisodes: " + str(self.totalEpisodes))
        print("Loaded model from disk")

######################################## MAIN #########################################################


if __name__ == "__main__":

    agent = Singleton("C:/Users/Tobias/Desktop/Master-Thesis/Data/EmoDB/wav/", "03a01Fa.wav", 7, 30, 30)

    if LOAD_PROGRESS_FROM_MODEL:
        agent.loadModel()

    data, sampling_rate = librosa.load(agent.wav_file_location + agent.wav_filename)
    agent.display_Data(data)

    data = agent.data_prep()
    print(data)

    if SAVE_PROGRESS_TO_MODEL:
        agent.saveModel()

## file: _models.py
## intended as a support file for _code.py
##

import numpy as np
import tensorflow as tf

# Import Keras Modules
from keras_vggface.vggface import VGGFace
from keras.layers import Dense, Flatten, Input, Dropout, Conv1D, Conv2D, LSTM, Concatenate, Reshape, MaxPool1D, MaxPool2D, BatchNormalization
from keras import Model, Sequential
import keras as keras

sess2 = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
tf.compat.v1.keras.backend.set_session(sess2)

def custom_vgg_model(is_trainable, combined_images, layer_regularization, regression, LSTM_layer):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(sess2)
        
        if combined_images == False and layer_regularization == False:
            model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
                        
            if is_trainable == False:
                for layer in model_VGGFace.layers:
                    layer.trainable = False
            else:
                for layer in model_VGGFace.layers:
                    layer.trainable = True

            last_layer = model_VGGFace.get_layer('avg_pool').output    
            print(last_layer.shape)

            if LSTM_layer == True:
                x = Reshape((1, last_layer.shape[0], last_layer.shape[1]))(last_layer)
                x = LSTM(16)(x)
                x = Flatten(name='flatten')(x)
            else:
                x = Flatten(name='flatten')(last_layer)
            
            x = Dropout(0.7)(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.6)(x)
            x = BatchNormalization()(x)

            if regression == True:
                out1 = Dense(1, activation='tanh', name='out1')(x)
                out2 = Dense(1, activation='tanh', name='out2')(x)
            else:
                out1 = Dense(21, activation='softmax', name='out1')(x)
                out2 = Dense(21, activation='softmax', name='out2')(x)
            custom_vgg_model = Model(inputs= model_VGGFace.input, outputs= [out1, out2])
            return custom_vgg_model


        elif combined_images == False and layer_regularization == True:
            model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

            regularizer = keras.regularizers.l1(0.01)
            if is_trainable == False:
                for layer in model_VGGFace.layers:
                    layer.trainable = False
            else:
                for layer in model_VGGFace.layers:
                    layer.trainable = True
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)
                model_VGGFace.save_weights("model_checkpoints/VGGFace_Regularization.h5")
                model_json = model_VGGFace.to_json()
                model_VGGFace = keras.models.model_from_json(model_json)
                model_VGGFace.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)
                
                # for layer in model_VGGFace.layers[:-25]:   ## all layers except the last .. layers
                #     layer.trainable = False

            last_layer = model_VGGFace.get_layer('avg_pool').output  

            if LSTM_layer == True:
                x = Reshape((last_layer.shape[1], 2048))(last_layer)
                x = LSTM(16)(x)
            else:
                x = Flatten(name='flatten')(last_layer)
            
            x = Dropout(0.7)(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.6)(x)
            x = BatchNormalization()(x)
            
            if regression == True:
                out1 = Dense(1, activation='tanh', name='out1')(x)
                out2 = Dense(1, activation='tanh', name='out2')(x)
            else:
                out1 = Dense(21, activation='softmax', name='out1')(x)
                out2 = Dense(21, activation='softmax', name='out2')(x)

            custom_vgg_model = Model(inputs= model_VGGFace.input, outputs= [out1, out2])
            return custom_vgg_model


        elif combined_images == True and layer_regularization == False:
            return 0


        else:
            model_VGGFace_1 = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            model_VGGFace_2 = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

            regularizer = keras.regularizers.l2(0.01)
            if is_trainable == False:
                for layer in model_VGGFace_1.layers:
                    layer.trainable = False
                for layer in model_VGGFace_2.layers:
                    layer.trainable = False
            else:
                for layer in model_VGGFace_1.layers:
                    layer.trainable = True
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)
                model_VGGFace_1.save_weights("model_checkpoints/VGGFace_Regularization.h5")
                model_json = model_VGGFace_1.to_json()
                model_VGGFace_1 = keras.models.model_from_json(model_json)
                model_VGGFace_1.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)

                for layer in model_VGGFace_2.layers:
                    layer.trainable = True
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)
                model_VGGFace_2.save_weights("model_checkpoints/VGGFace_Regularization.h5")
                model_json = model_VGGFace_2.to_json()
                model_VGGFace_2 = keras.models.model_from_json(model_json)
                model_VGGFace_2.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)

            for layer in model_VGGFace_2.layers:
                layer.name = str(layer.name) + "_2"

            last_layer_1 = model_VGGFace_1.get_layer('avg_pool').output
            x_1 = Flatten(name='flatten')(last_layer_1)
            last_layer_2 = model_VGGFace_2.get_layer('avg_pool_2').output  
            x_2 = Flatten(name='flatten_2')(last_layer_2)  
            x = Concatenate(axis=-1)([x_1, x_2])

            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = BatchNormalization()(x)
            out1 = Dense(21, activation='softmax', name='out1')(x)
            out2 = Dense(21, activation='softmax', name='out2')(x)
            custom_vgg_model = Model(inputs= [model_VGGFace_1.input, model_VGGFace_2.input], outputs= [out1, out2])
            return custom_vgg_model


def custom_vgg_model_w_landmarks(is_trainable, combined_images, layer_regularization):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(sess2)
        
        if combined_images == False and layer_regularization == False:
            model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            # model_VGGFace.trainable = True
            # for layer in model_VGGFace.layers[:-10]:   ## all layers except the last .. layers
            #     layer.trainable = False
            regularizer = keras.regularizers.l2(0.01)
            for layer in model_VGGFace.layers:
                layer.trainable = is_trainable
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)
            model_VGGFace.save_weights("model_checkpoints/VGGFace_Regularization.h5")
            model_json = model_VGGFace.to_json()
            model_VGGFace = keras.models.model_from_json(model_json)
            model_VGGFace.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)

            last_layer = model_VGGFace.get_layer('avg_pool').output    
            x = Flatten(name='flatten')(last_layer)
            landmarks_input = Input(shape=(136,))

            x = Concatenate(axis=-1)([x, landmarks_input])
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = BatchNormalization()(x)
            out1 = Dense(1, activation='tanh', name='out1')(x)
            out2 = Dense(1, activation='tanh', name='out2')(x)
            custom_vgg_model = Model(inputs= [model_VGGFace.input, landmarks_input] , outputs= [out1, out2])
            return custom_vgg_model

        elif combined_images == False and layer_regularization == True:
            model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            for layer in model_VGGFace.layers:
                layer.trainable = is_trainable

            last_layer = model_VGGFace.get_layer('avg_pool').output    
            x = Flatten(name='flatten')(last_layer)
            landmarks_input = Input(shape=(136,))

            x = Concatenate(axis=-1)([x, landmarks_input])
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = BatchNormalization()(x)
            out1 = Dense(1, activation='tanh', name='out1')(x)
            out2 = Dense(1, activation='tanh', name='out2')(x)
            custom_vgg_model = Model(inputs= [model_VGGFace.input, landmarks_input] , outputs= [out1, out2])
            return custom_vgg_model

def custom_landmarks_model():
    landmarks_input = Input(shape=(136,))
    
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)

    x = Dense(4096, activation='relu')(landmarks_input)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    out1 = Dense(1, activation='tanh', name='out1')(x)
    out2 = Dense(1, activation='tanh', name='out2')(x)
    custom_landmarks_model = Model(inputs=landmarks_input , outputs= [out1, out2])
    
    return custom_landmarks_model


def custom_model(is_trainable):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(sess2)
 
        model_input = Input(shape=(224, 224, 3))
        x = Conv2D(64, (11, 11), activation='relu')(model_input)
        x = Conv2D(64, (11, 11), activation='relu')(x)
        x = Conv2D(64, (11, 11), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Conv2D(256, (5, 5), activation='relu')(x)
        x = Conv2D(256, (5, 5), activation='relu')(x)
        x = Conv2D(256, (5, 5), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        x = Flatten(name='flatten')(x)    
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        out1 = Dense(1, activation='tanh', name='out1')(x)
        out2 = Dense(1, activation='tanh', name='out2')(x)
        custom_vgg_model = Model(inputs=model_input , outputs= [out1, out2])

        return custom_vgg_model

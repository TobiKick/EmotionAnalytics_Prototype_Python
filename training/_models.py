## file: _models.py
## intended as a support file for _code.py
##

import numpy as np
import tensorflow as tf

# Import Keras Modules
from keras_vggface.vggface import VGGFace
from keras.layers import Add, Activation, Dense, Flatten, Input, Dropout, Conv1D, Conv2D, LSTM, Concatenate, Reshape, MaxPool1D, MaxPool2D, BatchNormalization, TimeDistributed, Reshape, GlobalAveragePooling2D
from keras import Model, Sequential
from keras import activations
import keras as keras
import keras.backend as K

# sess2 = tf.compat.v1.Session()
# graph = tf.compat.v1.get_default_graph()
# K.set_session(sess2)

RESNET = 18

def custom_vgg_model(is_trainable, conv_block, layer_regularization, model_option):
    # global sess2
    # global graph
    # with graph.as_default():
    #     sess2 = tf.compat.v1.Session()
    #     graph = tf.compat.v1.get_default_graph()
    # K.set_session(sess2)

    if model_option == 4: # self coded ResNet50
        if RESNET == 18:
            model_input = Input(shape=(224, 224, 4))
            x = Conv2D(filters=64, kernel_size=(7, 7), strides=2)(model_input)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x1 = MaxPool2D((3,3), strides=2, padding="same")(x)

            # Stage 1
            for i in [0, 1]:
                x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x1)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x)
                x2 = BatchNormalization()(x)
                x = Add()([x1, x2])
                x1 = Activation(activations.relu)(x)

            # Stage 2
            # Conv block
            x = Conv2D(filters=128, kernel_size=(3,3), strides=2, padding='same')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(x)
            x2 = BatchNormalization()(x)
            
            x1 = Conv2D(filters=128, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 2
            # Identity block
            x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(x)
            x2 = BatchNormalization()(x)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 3
            # Conv block
            x = Conv2D(filters=256, kernel_size=(3,3), strides=2, padding='same')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(x)
            x2 = BatchNormalization()(x)
            
            x1 = Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 3
            # Identity block
            x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(x)
            x2 = BatchNormalization()(x)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 4
            # Conv block
            x = Conv2D(filters=512, kernel_size=(3,3), strides=2, padding='same')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(x)
            x2 = BatchNormalization()(x)
            
            x1 = Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 4
            # Identity block
            x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(x)
            x2 = BatchNormalization()(x)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)


        elif RESNET == 50:
            model_input = Input(shape=(224, 224, 4))
            x = Conv2D(filters=64, kernel_size=(7, 7), strides=2)(model_input)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x1 = MaxPool2D((3,3), strides=2, padding="same")(x)

            # Stage 1
            # Conv block
            x = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='valid')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='valid')(x)
            x2 = BatchNormalization()(x)
            
            x1 = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='valid')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 1
            # Identity block
            for i in [0, 1]:
                x = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='valid')(x1)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='valid')(x)
                x2 = BatchNormalization()(x)
                x = Add()([x1, x2])
                x1 = Activation(activations.relu)(x)

            # Stage 2
            # Conv block
            x = Conv2D(filters=128, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=512, kernel_size=(1,1), strides=1, padding='valid')(x)
            x2 = BatchNormalization()(x)
            
            x1 = Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 2
            # Identity block
            for i in [0, 1, 2]:
                x = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='valid')(x1)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=512, kernel_size=(1,1), strides=1, padding='valid')(x)
                x2 = BatchNormalization()(x)
                x = Add()([x1, x2])
                x1 = Activation(activations.relu)(x)

            # Stage 3
            # Conv block
            x = Conv2D(filters=256, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=1024, kernel_size=(1,1), strides=1, padding='valid')(x)
            x2 = BatchNormalization()(x)
            
            x1 = Conv2D(filters=1024, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 3
            # Identity block
            for i in [0, 1, 2, 3, 4]:
                x = Conv2D(filters=256, kernel_size=(1,1), strides=1, padding='valid')(x1)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=1024, kernel_size=(1,1), strides=1, padding='valid')(x)
                x2 = BatchNormalization()(x)
                x = Add()([x1, x2])
                x1 = Activation(activations.relu)(x)

            # Stage 4
            # Conv block
            x = Conv2D(filters=512, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activations.relu)(x)
            x = Conv2D(filters=2048, kernel_size=(1,1), strides=1, padding='valid')(x)
            x2 = BatchNormalization()(x)
            
            x1 = Conv2D(filters=2048, kernel_size=(1,1), strides=2, padding='valid')(x1)
            x1 = BatchNormalization()(x1)
            x = Add()([x1, x2])
            x1 = Activation(activations.relu)(x)

            # Stage 4
            # Identity block
            for i in [0, 1]:
                x = Conv2D(filters=512, kernel_size=(1,1), strides=1, padding='valid')(x1)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation(activations.relu)(x)
                x = Conv2D(filters=2048, kernel_size=(1,1), strides=1, padding='valid')(x)
                x2 = BatchNormalization()(x)
                x = Add()([x1, x2])
                x1 = Activation(activations.relu)(x)


        x = GlobalAveragePooling2D()(x1)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)

        out1 = Dense(1, activation='tanh', name='out1')(x)
        out2 = Dense(1, activation='tanh', name='out2')(x)

        custom_vgg_model = Model(inputs=model_input, outputs= [out1, out2])
        return custom_vgg_model
    
    
    elif model_option == 3: # with mask - 4 channel input with VGGFace
        
        model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 4), pooling='avg')

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

        last_layer = model_VGGFace.get_layer('avg_pool').output    
        print(last_layer.shape)
        x = Flatten(name='flatten')(last_layer)

        x = Dropout(0.7)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = BatchNormalization()(x)

        out1 = Dense(1, activation='tanh', name='out1')(x)
        out2 = Dense(1, activation='tanh', name='out2')(x)

        custom_vgg_model = Model(inputs=model_VGGFace, outputs= [out1, out2])
        return custom_vgg_model



    if model_option == 2: # with mask
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

        last_layer = model_VGGFace.get_layer('avg_pool').output    
        print(last_layer.shape)
        x = Flatten(name='flatten')(last_layer)
        #######
        #######
        y_input = Input(shape=(224, 224))
        y = Reshape((224, 224, 1), input_shape=(224,224))(y_input)

        y = Conv2D(filters=32, kernel_size=(7,7), strides=2, activation='relu')(y)
        y = MaxPool2D((2,2), padding="valid")(y)

        y = Conv2D(filters=32, kernel_size=(3,3), strides=1, activation='relu')(y)
        y = MaxPool2D((2,2), padding="valid")(y)

        y = Conv2D(filters=64, kernel_size=(3,3), strides=2, activation='relu')(y)
        y = Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu')(y)
        y = MaxPool2D((2,2), padding="valid")(y)
        y = BatchNormalization()(y)
        y = Flatten()(y)
        #######
        #######
        xy = Concatenate(axis=-1)([x, y])
        xy = Dropout(0.7)(xy)
        xy = Dense(1024, activation='relu')(xy)
        xy = Dropout(0.6)(xy)
        xy = BatchNormalization()(xy)

        out1 = Dense(1, activation='tanh', name='out1')(xy)
        out2 = Dense(1, activation='tanh', name='out2')(xy)

        custom_vgg_model = Model(inputs=[model_VGGFace.input, y_input], outputs= [out1, out2])
        return custom_vgg_model


    elif model_option == 1: # with LSTM
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

        model = Sequential()
        model.add(TimeDistributed(model_VGGFace, input_shape=(5, 224, 224, 3)))
        model.add(TimeDistributed(Flatten(), name='flatten'))
        model.add(LSTM(256, activation='relu', return_sequences=False, name='lstm'))
        model.add(Dropout(0.7, name='dropout'))
        model.add(Dense(1024, activation='relu', name='dense'))
        model.add(Dropout(0.6, name='dropout2'))
        model.add(BatchNormalization(name='batchNorm'))
        model.add(Dense(2, activation='tanh', name='out'))
        return model


    elif layer_regularization == False:
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

        last_layer = model_VGGFace.get_layer('avg_pool').output    
        print(last_layer.shape)

        x = Flatten(name='flatten')(last_layer)
        x = Dropout(0.7)(x)
        # x = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = BatchNormalization()(x)

        out1 = Dense(1, activation='tanh', name='out1')(x)
        out2 = Dense(1, activation='tanh', name='out2')(x)

        custom_vgg_model = Model(inputs= model_VGGFace.input, outputs= [out1, out2])
        return custom_vgg_model


    else:
        model_VGGFace = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        regularizer = keras.regularizers.l2(0.001)
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

                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)

            model_VGGFace.save_weights("model_checkpoints/VGGFace_Regularization.h5")
            model_json = model_VGGFace.to_json()
            model_VGGFace = keras.models.model_from_json(model_json)
            model_VGGFace.load_weights("model_checkpoints/VGGFace_Regularization.h5", by_name=True)
            

        last_layer = model_VGGFace.get_layer('avg_pool').output  

        x = Flatten(name='flatten')(last_layer)
        x = Dropout(0.7)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = BatchNormalization()(x)
        
        out1 = Dense(1, activation='tanh', name='out1')(x)
        out2 = Dense(1, activation='tanh', name='out2')(x)

        custom_vgg_model = Model(inputs= model_VGGFace.input, outputs= [out1, out2])
        return custom_vgg_model


def custom_vgg_model_w_landmarks(is_trainable, layer_regularization):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        K.set_session(sess2)
        
        if layer_regularization == False:
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

        else:
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

def custom_model(is_trainable):
    global sess2
    global graph
    with graph.as_default():
        sess2 = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        K.set_session(sess2)
 
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

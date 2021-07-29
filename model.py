import os
import numpy as np
import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers import Input, ReLU, Conv2DTranspose, Reshape
from keras.regularizers import l2
from params import *


def load_model(file_path,model,procedure):
    if not os.path.exists(file_path):
        if model == 'encoder':
            return create_encoder(procedure)
        elif model == 'decoder':
            return create_decoder()
        elif model == 'classifier':
            return create_classifier()
        else :
            print("/!\ Unknown model /!\ ")
            exit(0)
    else :
        return keras.models.load_model(file_path)


def create_encoder(procedure):
    '''Create the model for the encoder'''
    input = Input(shape=INPUT_SHAPE)

    if procedure: #We put no dropout when training the autoencoder
        x = layers.Conv2D(NB_FILTER1, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input)
        # x = keras.layers.Dropout(0.2) (x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(NB_FILTER2, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        # x = keras.layers.Dropout(0.2) (x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(NB_FILTER3, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    else: #We use dropout when training the classifier
        x = layers.Conv2D(NB_FILTER1, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input)
        x = keras.layers.Dropout(0.2) (x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(NB_FILTER2, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        x = keras.layers.Dropout(0.2) (x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(NB_FILTER3, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        

    encoder = Model(input, encoded, name='encoder')

    return encoder


def create_decoder():
    '''Create the model for the decoder'''
    input = Input(shape=DIM_LATENT)

    x = layers.Conv2D(NB_FILTER3, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(new_input)
    # x = keras.layers.Dropout(0.2) (x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(NB_FILTER2, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    # x = keras.layers.Dropout(0.2) (x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(NB_FILTER1, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    # x = keras.layers.Dropout(0.2) (x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=l2(0.001))(x)


    decoder = Model(input, decoded, name='decoder')

    return decoder


def assemble_autoencoder(encoder, decoder):
    '''Put the encoder and decoder together to create the autoencoder'''
    input = Input(shape=INPUT_SHAPE)
    autoencoder = Model(input, decoder(encoder(input)), name='autoencoder')

    def loss_fn(y_true, y_pred):
        return keras.losses.MeanSquaredError(reduction='auto')(y_true,y_pred) + BETA*keras.losses.BinaryCrossentropy(reduction='auto')(y_true,y_pred)
    
    
    autoencoder.compile(loss=loss_fn,metrics=['mse','binary_crossentropy'], optimizer='adam')

    return autoencoder


def create_classifier():
    '''Create classifier in two parts; the first part has the same architecture as the encoder'''
    continuation_classifier = Input(shape=(DIM_LATENT,))

    x = Dense(NB_NEURON1, activation='relu', kernel_regularizer=l2(0.001)) (continuation_classifier)
    x = keras.layers.Dropout(0.2) (x)
    x = Dense(NB_NEURON2, activation='relu', kernel_regularizer=l2(0.001)) (x)
    x = keras.layers.Dropout(0.2) (x)
    x = Dense(NB_NEURON3, activation='relu', kernel_regularizer=l2(0.001)) (x)
    x = keras.layers.Dropout(0.2) (x)
    x = Dense(NB_CLASSES, activation='softmax', name='classifier_output', kernel_regularizer=l2(0.001)) (x)

    classifier = Model(continuation_classifier, x, name='classifier_end')


    classifier.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    return classifier


def assemble_classifier(classifier_beginning, classifier_end):
    '''Put the two parts of the classifier together to create the classifier'''
    input = Input(shape=INPUT_SHAPE)

    classifier = Model(input, classifier_end(classifier_beginning(input)), name='classifier')


    classifier.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    return classifier


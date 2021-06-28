import os
import numpy as np
import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers import Input, ReLU, Conv2DTranspose, Reshape
from params import INPUT_SHAPE, DIM_LATENT, NB_CLASSES, OPTIMIZER, BETA, NB_FILTER1, NB_FILTER2


def load_model(file_path,model):
    if not os.path.exists(file_path):
        if model == 'encoder':
            return create_encoder()
        elif model == 'decoder':
            return create_decoder()
        elif model == 'classifier':
            return create_classifier()
        else :
            print("/!\ Unknown model /!\ ")
            exit(0)
    else :
        return keras.models.load_model(file_path)


def create_encoder():
    '''Create the model for the encoder'''
    input = Input(shape=INPUT_SHAPE)

    x = layers.Conv2D(NB_FILTER1, (3, 3), activation='relu', padding='same')(input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(NB_FILTER2, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = Model(input, encoded, name='encoder')

    return encoder


def create_decoder():
    '''Create the model for the decoder'''
    input = Input(shape=DIM_LATENT)

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(NB_FILTER2, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(NB_FILTER1, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(input, decoded, name='decoder')

    return decoder


def assemble_autoencoder(encoder, decoder):
    '''Put the encoder and decoder together to create the autoencoder'''
    input = Input(shape=INPUT_SHAPE)

    autoencoder = Model(input, decoder(encoder(input)), name='autoencoder')

    autoencoder.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)

    return autoencoder


def create_classifier():
    '''Create classifier in two parts; the first part has the same architecture as the encoder'''
    continuation_classifier = Input(shape=(DIM_LATENT,))

    classif2 = Dense(64, activation='relu') (continuation_classifier)
    classif3 = Dense(32, activation='relu') (classif2)
    classif4 = Dense(16, activation='relu') (classif3)
    classif5 = Dense(NB_CLASSES, activation='softmax', name='classifier_output') (classif4)

    classifier = Model(continuation_classifier, classif5, name='classifier_end')
    classifier.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    return classifier


#def assemble_classifier(classifier_beginning, classifier_end):
    #'''Put the two parts of the classifier together to create the classifier'''
    #input = Input(shape=INPUT_SHAPE)

    #classifier = Model(input, classifier_end(classifier_beginning(input)), name='classifier')

    #classifier.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    #return classifier


def assemble_fusion(encoder, decoder, classifier_end):

    input_fusion = Input(shape=INPUT_SHAPE)

    fusion = Model(input_fusion, [ decoder(encoder(input_fusion)), classifier_end(encoder(input_fusion)) ], name='fusion')
    
    fusion.compile(optimizer=OPTIMIZER, 
              loss={
                  'decoder': 'mse', 
                  'classifier_end': 'categorical_crossentropy'},
              loss_weights={
                  'decoder': 1., 
                  'classifier_end': BETA},
              metrics={ 
                  'classifier_end': 'accuracy'})
        
    return fusion


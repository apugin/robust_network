import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers import Input, LeakyReLU, Conv2DTranspose, Reshape
from params import INPUT_SHAPE, DIM_LATENT, ALPHA, NB_CLASSES


def create_encoder():
    '''Create the model for the encoder'''
    input = Input(shape=INPUT_SHAPE)

    encoder2 = Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same') (input)
    encoder3 = LeakyReLU(alpha=ALPHA) (encoder2)

    encoder4 = Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same') (encoder3)
    encoder5 = LeakyReLU(alpha=ALPHA) (encoder4)

    encoder6 = Flatten() (encoder5)
    encoder7 = Dense(DIM_LATENT) (encoder6)

    encoder = Model(input, encoder7, name='encoder')

    return encoder


def create_decoder(encoder):
    '''Create the model for the decoder'''
    volume_size = encoder.layers[-2].input_shape

    decoder1 = Input(shape=(DIM_LATENT,))

    decoder2 = Dense(np.prod(volume_size[1:])) (decoder1)
    decoder3 = Reshape((volume_size[1], volume_size[2], volume_size[3])) (decoder2)

    decoder4 = Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='same') (decoder3)
    decoder5 = LeakyReLU(alpha=ALPHA) (decoder4)

    decoder6 = Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='same') (decoder5)
    decoder7 = LeakyReLU(alpha=ALPHA) (decoder6)

    decoder8 = Conv2DTranspose(INPUT_SHAPE[2], kernel_size=(3,3), padding='same', activation='sigmoid',name='decoder_output') (decoder7)

    decoder = Model(decoder1, decoder8, name='decoder')

    return decoder


def assemble_autoencoder(encoder, decoder):
    '''Put the encoder and decoder together to create the autoencoder'''
    input = Input(shape=INPUT_SHAPE)

    autoencoder = Model(input, decoder(encoder(input)), name='autoencoder')

    return autoencoder


def create_classifier_two_parts():
    '''Create classifier in two parts; the first part has the same architecture as the encoder'''
    classifier_beginning = create_encoder()

    continuation_classifier = Input(shape=(DIM_LATENT,))

    end_classif2 = Dense(NB_CLASSES, activation='sigmoid', name='classifier_output') (continuation_classifier)

    classifier_end = Model(continuation_classifier, end_classif2, name='classifier_end')

    return classifier_beginning, classifier_end


def assemble_classifier(classifier_beginning, classifier_end):
    '''Put the two parts of the classifier together to create the classifier'''
    input = Input(shape=INPUT_SHAPE)

    classifier = Model(input, classifier_end(classifier_beginning(input)), name='classifier')

    return classifier

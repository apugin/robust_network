import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, LeakyReLU, Conv2DTranspose, Reshape, ReLU
from kerastuner import HyperModel
from params import NB_CLASSES, INPUT_SHAPE


class AEHyperModel(HyperModel):
  def build(self, hp):
    nb_filters1 = hp.Int(
        'nb_filters1',
        min_value=32,
        max_value=128,
        step=32,
        default=64
    )

    #alpha = hp.Float(
    #                'alpha',
    #                min_value=1e-3,
    #                max_value=1,
    #                sampling='LOG',
    #                default=1e-1
    #            )

    #activation_choice = hp.Choice(
    #    'activation_choice',
    #    values=['relu','leakyrelu']
    #)

    filter_size = hp.Int(
        'filter_size',
        min_value=2,
        max_value=6,
        step=1,
        default=3
    )

    nb_filters2 = hp.Int(
        'nb_filters2',
        min_value=16,
        max_value=64,
        step=16,
        default=32
    )

    dim_latent = hp.Int(
        'dim_latent',
        min_value=10,
        max_value=16,
        step=3,
        default=10
    )

    #learning_rate = hp.Float(
    #                'learning_rate',
    #                min_value=1e-4,
    #                max_value=1e-2,
    #                sampling='LOG',
    #                default=1e-3
    #                )

    create_autoencoder(nb_filters1, nb_filters2, filter_size, dim_latent)

    return autoencoder

def create_autoencoder(nb_filters1, nb_filters2, filter_size, dim_latent):
  autoencoder = keras.Sequential()

  activation = ReLU(max_value=None, negative_slope=0, threshold=0)

  autoencoder.add(Conv2D(nb_filters1, kernel_size=filter_size, strides=(2,2), padding='same', input_shape=INPUT_SHAPE))
  autoencoder.add(activation)

  autoencoder.add(Conv2D(nb_filters2, kernel_size=filter_size, strides=(2,2), padding='same'))
  autoencoder.add(activation)

  autoencoder.add(Flatten())
  autoencoder.add(Dense(dim_latent))

  volume_size = (None, 7, 7, nb_filters2)

  autoencoder.add(Dense(np.prod(volume_size[1:])))
  autoencoder.add(Reshape((volume_size[1], volume_size[2], volume_size[3])))

  autoencoder.add(Conv2DTranspose(nb_filters2, kernel_size=filter_size, strides=(2,2), padding='same'))
  autoencoder.add(activation)

  autoencoder.add(Conv2DTranspose(nb_filters1, kernel_size=filter_size, strides=(2,2), padding='same'))
  autoencoder.add(activation)

  autoencoder.add(Conv2DTranspose(INPUT_SHAPE[2], kernel_size=filter_size, padding='same', activation='sigmoid'))

  learning_rate = 1e-3

  autoencoder.compile(
          optimizer=keras.optimizers.Adam(
              learning_rate    
          ),
          loss='mse'
      )
    
  return autoencoder

def sklearn_autoencoder():
    
    autoencoder = KerasRegressor(build_fn=create_autoencoder, verbose=0)

    return autoencoder


def create_classifier(nb_filters1, nb_filters2, filter_size, dim_latent, nb_layers, nb_neurons):
  classifier = keras.Sequential()

  activation = ReLU(max_value=None, negative_slope=0, threshold=0)

  classifier.add(Conv2D(nb_filters1, kernel_size=filter_size, strides=(2,2), padding='same', input_shape=INPUT_SHAPE))
  classifier.add(activation)

  classifier.add(Conv2D(nb_filters2, kernel_size=filter_size, strides=(2,2), padding='same'))
  classifier.add(activation)

  classifier.add(Flatten())
  classifier.add(Dense(dim_latent))

  for i in range(nb_layers):
      classifier.add(Dense(nb_neurons))
      classifier.add(activation)

  learning_rate = 1e-3

  classifier.add(Dense(NB_CLASSES, activation='softmax', name='classifier_output'))

  classifier.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
    
  return classifier


def sklearn_classifier():
    
    classifier = KerasClassifier(build_fn=create_classifier, verbose=0)

    return classifier


def load_fusion(beta):
    encoder = keras.models.load_model("saved_models/encoder_1.0.h5")
    decoder = keras.models.load_model("saved_models/decoder_1.0.h5")
    classifier_end = keras.models.load_model("saved_models/classifier_end_1.0.h5")

    input_fusion = Input(shape=INPUT_SHAPE)

    learning_rate = 1e-3

    fusion = Model(input_fusion, [ decoder(encoder(input_fusion)), classifier_end(encoder(input_fusion)) ], name='fusion')
    fusion.compile(optimizer=keras.optimizers.Adam(learning_rate), 
            loss={
                'decoder': 'mse', 
                'classifier_end': 'categorical_crossentropy'},
            loss_weights={
                'decoder': 1., 
                'classifier_end': beta},
            metrics={'classifier_end': 'accuracy'})
    
    return fusion
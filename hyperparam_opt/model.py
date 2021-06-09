from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, LeakyReLU, Conv2DTranspose, Reshape, ReLU
from kerastuner import HyperModel
from params import INPUT_SHAPE, NB_CLASSES, BETA


class AEHyperModel(HyperModel):
  def __init__(self, input_shape):
    self.input_shape = input_shape

  def build(self, hp):
    autoencoder = keras.Sequential()

    nb_filters1 = hp.Int(
        'nb_filters1',
        min_value=32,
        max_value=128,
        step=32,
        default=64
    )

    alpha = hp.Float(
                    'alpha',
                    min_value=1e-3,
                    max_value=1,
                    sampling='LOG',
                    default=1e-1
                )

    activation_choice = hp.Choice(
        'activation_choice',
        values=['relu','leakyrelu']
    )

    if activation_choice == 'relu':
      activation = ReLU(max_value=None, negative_slope=0, threshold=0)
    elif activation_choice == 'leakyrelu':
      activation = LeakyReLU(alpha=alpha)

    filter_size = hp.Int(
        'filter_size',
        min_value=2,
        max_value=6,
        step=1,
        default=3
    )

    autoencoder.add(Conv2D(nb_filters1, kernel_size=filter_size, strides=(2,2), padding='same', input_shape=input_shape))
    autoencoder.add(activation)

    nb_filters2 = hp.Int(
        'nb_filters2',
        min_value=16,
        max_value=64,
        step=16,
        default=32
    )

    autoencoder.add(Conv2D(nb_filters2, kernel_size=filter_size, strides=(2,2), padding='same'))
    autoencoder.add(activation)

    dim_latent = hp.Int(
        'dim_latent',
        min_value=8,
        max_value=16,
        step=2,
        default=10
    )

    autoencoder.add(Flatten())
    autoencoder.add(Dense(dim_latent))

    # On construit le décodeur de façon symétrique à l'encodeur

    volume_size = (None, 7, 7, nb_filters2)

    autoencoder.add(Dense(np.prod(volume_size[1:])))
    autoencoder.add(Reshape((volume_size[1], volume_size[2], volume_size[3])))

    autoencoder.add(Conv2DTranspose(nb_filters2, kernel_size=filter_size, strides=(2,2), padding='same'))
    autoencoder.add(activation)

    autoencoder.add(Conv2DTranspose(nb_filters1, kernel_size=filter_size, strides=(2,2), padding='same'))
    autoencoder.add(activation)

    autoencoder.add(Conv2DTranspose(input_shape[2], kernel_size=filter_size, padding='same', activation='sigmoid'))

    autoencoder.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )    
            ),
            loss='mse'
        )
    
    return autoencoder


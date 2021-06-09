from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from params import IMG_ROWS, IMG_COLS, NB_CLASSES


def get_data(nb_training_samples):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.

    nb_samples = len(x_train)
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

    l_idx = [i for i in range(nb_samples)]
    np.random.shuffle(l_idx)
    l_idx = l_idx[:nb_training_samples]
    x_train, y_train = x_train[l_idx], y_train[l_idx]

    y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return (x_train, y_train), (x_test, y_test)
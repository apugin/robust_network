from keras.datasets import mnist
from params import IMG_ROWS, IMG_COLS


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

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return (x_train, y_train), (x_test, y_test)
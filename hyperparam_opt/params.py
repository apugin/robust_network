from keras.optimizers import SGD, Adam


# Data parameters
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)


# Training parameters
BETA = 1 # Parameter for loss function: Crossentropy(labels) + BETA*MSE(images)


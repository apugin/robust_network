from keras.optimizers import SGD, Adam


# Data parameters
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)


# Training parameters
TAUX_APPRENTISAGE = 0.01
OPTIMIZER = 'adam'
BETA = 1 # Parameter for loss function: Crossentropy(labels) + BETA*MSE(images)


# Model parameters
ALPHA = 0.2 # Parameter for LeakyReLu
DIM_LATENT = 16 # Dimension of the latent space


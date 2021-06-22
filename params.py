from keras.optimizers import SGD, Adam


# Data parameters
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)


# Training parameters
TAUX_APPRENTISAGE = 1e-3
OPTIMIZER = Adam(TAUX_APPRENTISAGE)
BETA = 10**(-0.84375) # Parameter for loss function: BETA*Crossentropy(labels) + MSE(images)


# Model parameters
NB_FILTER1 = 128
NB_FILTER2 = 64
DIM_LATENT = (4,4,8) # Dimension of the latent space

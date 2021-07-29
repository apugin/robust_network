from keras.optimizers import SGD, Adam


# Data parameters
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NB_CL = 9 # Number of random CLs generated for each vector when augmenting data set
K = 3 # Number of nearest neighbors of the same class used for random Cls


# Training parameters
VAL_SPLIT = 0.2
OPTIMIZER = 'adam'
BETA = 10 # Parameter for loss function: BETA*Binary_Crossentropy(images) + MSE(images)


# Model parameters
#For encoder/decoder
NB_FILTER1 = 128
NB_FILTER2 = 64
NB_FILTER3 = 8
DIM_LATENT = 4*4*NB_FILTER3 # Dimension of the latent space (DO NOT CHANGE IT/!\)

#For classifier
NB_NEURON1 = 96
NB_NEURON2 = 32
NB_NEURON3 = 32

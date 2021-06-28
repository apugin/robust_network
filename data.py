import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Flatten, Input
from keras.models import Model
import numpy as np
from sklearn.neighbors import NearestNeighbors
from params import IMG_ROWS, IMG_COLS, NB_CLASSES, INPUT_SHAPE, NB_CL, DIM_LATENT


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


def get_h_data(x_data):
    encoder = keras.models.load_model("saved_models/encoder_2.0.h5")

    input = Input(shape=(4,4,8))
    x = Flatten() (input)
    flattener = Model(input, x)

    enc_input = Input(shape=INPUT_SHAPE)
    flat_encoder = Model(enc_input, flattener(encoder(enc_input)))
    h_data = flat_encoder.predict(x_data)
    return h_data


def k_neighbors_graph(data_set, k):
  nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_set)
  graph = nbrs.kneighbors_graph(data_set).toarray()
  return graph


def sort_by_class(h_train, y_train):
  data_sorted = [[] for k in range(NB_CLASSES)]
  y_train_num = np.argmax(y_train,axis=1)
  for i in range(len(y_train)):
    classe = y_train_num[i]
    data_sorted[classe].append(h_train[i])
  for k in range(NB_CLASSES):
    data_sorted[k] = np.array(data_sorted[k])
  return data_sorted


def random_cl(vector_list):
  n = len(vector_list)
  linear_combinations=[]
  for i in range(NB_CL):
    weights = np.random.random((n,))
    weights = weights/np.sum(weights, axis=0)
    cl = np.zeros((DIM_LATENT,))
    for i in range(n):
      cl += weights[i]*vector_list[i]
    linear_combinations.append(cl)
  return linear_combinations


def augmentation(h_train, y_train, k):
  augmented_data = []
  augmented_y = []
  data_sorted = sort_by_class(h_train, y_train)
  for i in range(NB_CLASSES):
    data = data_sorted[i]
    graph = k_neighbors_graph(data, k+1)
    for j in range(data.shape[0]):
      neighbors = []
      l = 0
      while len(neighbors) < k+1:
        if graph[j,l] == 1:
          neighbors.append(data[l,:])
        l += 1
      linear_combinations = random_cl(neighbors)
      for element in linear_combinations:
        augmented_data.append(element)
        augmented_y.append(i)
      augmented_data.append(data[j,:])
      augmented_y.append(i)

  augmented_data = np.array(augmented_data)
  augmented_y = np.array(augmented_y)
  augmented_y = np_utils.to_categorical(augmented_y, NB_CLASSES)

  l_idx = [i for i in range(augmented_data.shape[0])]
  np.random.shuffle(l_idx)
  l_idx = l_idx[:augmented_data.shape[0]]
  augmented_data, augmented_y = augmented_data[l_idx], augmented_y[l_idx]
  
  return augmented_data, augmented_y


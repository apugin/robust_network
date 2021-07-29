import keras
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import foolbox as fb
import matplotlib.pyplot as plt
from data import get_h_data
from model import *
from params import OPTIMIZER, INPUT_SHAPE


def testing(x_test,y_test,epsilons):
    #Looking up the name of the models saved
    relevant_path = "saved_models"
    file_names = [fn for fn in os.listdir(relevant_path)
              if fn.startswith("classifier_end")]

    models_list = []
    for fn in file_names:
        models_list.append(fn[15:-3]) #We extract the name given to the model
    
    #Foolbox metric
    #We put data in the right format to use foolbox
    X_test = K.constant(x_test[:1000])
    Y_test = K.constant(np.argmax(y_test[:1000], axis=-1))
    Y_test = tf.cast(Y_test, tf.int32)

    attack = fb.attacks.LinfFastGradientAttack()

    #We get the accuracy for each model depending of epsilon
    models_accuracy = []

    for model_name in models_list:
        encoder_name = "saved_models/encoder_" + model_name + ".h5"
        classifier_end_name = "saved_models/classifier_end_" + model_name + ".h5"

        encoder = keras.models.load_model(encoder_name)
        classifier_end = keras.models.load_model(classifier_end_name)

        enc_input = Input(shape=INPUT_SHAPE)
        classifier = Model(enc_input, classifier_end(Flatten()(encoder(enc_input))))
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        fmodel = fb.TensorFlowModel(classifier, bounds=(0, 1))

        _, advs, success = attack(fmodel, X_test, Y_test, epsilons=epsilons)
        success_ = tf.cast(success,tf.float32)
        robust_accuracy = 1 - tf.math.reduce_mean(success_,axis=1)
        models_accuracy.append(robust_accuracy.numpy())


    for i in range(len(models_list)):
        plt.plot(epsilons[:], models_accuracy[i][:])

    plt.legend(models_list)
    plt.xlabel("Epsilon")
    # plt.xlabel("L2 size of noise")
    plt.ylabel("Accuracy on testing set")
    plt.show()
 
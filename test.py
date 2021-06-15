import keras
from keras.layers import Input
from keras.models import Model
from model import *
from params import OPTIMIZER, INPUT_SHAPE, BETA


def testing(x_test,y_test,model,file):
    if model == 'autoencoder':
        encoder_path = "saved_models/encoder" + file + ".h5"
        decoder_path = "saved_models/decoder" + file + ".h5"

        encoder = load_model(encoder_path, "encoder")
        decoder = load_model(decoder_path, "decoder")

        autoencoder = assemble_autoencoder(encoder, decoder)

        score = autoencoder.evaluate(x_test, x_test, verbose=0)
        print("Accuracy de l'autoencodeur:"+str(score))


    elif model == 'classifier':
        classifier_beginning_path = "saved_models/classifier_beginning" + file + ".h5"
        classifier_end_path = "saved_models/classifier_end" + file + ".h5"

        classifier_beginning = load_model(classifier_beginning_path, "encoder")
        classifier_end = load_model(classifier_end_path, "classifier_end")

        classifier = assemble_classifier(classifier_beginning, classifier_end)

        score = classifier.evaluate(x_test, y_test, verbose=0)
        print("Accuracy du classifieur:"+str(score[1]))


    elif model == 'fusion':
        encoder_path = "saved_models/encoder" + file + ".h5"
        decoder_path = "saved_models/decoder" + file + ".h5"
        classifier_end_path = "saved_models/classifier_end" + file + ".h5"

        encoder = load_model(encoder_path, "encoder")
        decoder = load_model(decoder_path, "decoder")
        classifier_end = load_model(classifier_end_path, "classifier_end")

        fusion = assemble_fusion(encoder, decoder, classifier_end)
        
        score = fusion.evaluate(x=x_test,
	            y={"decoder": x_test, "classifier_end": y_test},
	            verbose=0)
        print("Accuracy du décodeur dans fusion:"+str(score[3]))
        print("Accuracy du classifieur dans fusion:"+str(score[4]))


    else:
        print("/!\ Unknown model : type 'autoencoder', 'classifier' or 'fusion'")
        exit(0)
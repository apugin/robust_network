import keras
from keras.layers import Input
from keras.models import Model
from params import OPTIMIZER, INPUT_SHAPE, BETA


def testing(x_test,y_test,model):
    if model == 'autoencoder':
        encoder = keras.models.load_model("saved_models/encoder.h5")
        decoder = keras.models.load_model("saved_models/decoder.h5")

        input = Input(shape=INPUT_SHAPE)
        autoencoder = Model(input, decoder(encoder(input)), name='autoencoder')
        autoencoder.compile(loss="mse", optimizer=OPTIMIZER, metrics=['accuracy'])

        score = autoencoder.evaluate(x_test, x_test, verbose=0)
        print("Accuracy de l'autoencodeur:"+str(score[1]))


    elif model == 'classifier':
        classifier_beginning = keras.models.load_model("saved_models/classifier_beginning.h5")
        classifier_end = keras.models.load_model("saved_models/classifier_end.h5")

        input_classifier = Input(shape=INPUT_SHAPE)
        classifier = Model(input_classifier, classifier_end(classifier_beginning(input_classifier)), name='classifier')
        classifier.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

        score == classifier.evaluate(x_test, y_test, verbose=0)
        print("Accuracy du classifieur:"+str(score[1]))


    elif model == 'fusion':
        encoder = keras.models.load_model("saved_models/encoder.h5")
        decoder = keras.models.load_model("saved_models/decoder.h5")
        classifier_end = keras.models.load_model("saved_models/classifier_end.h5")

        input_fusion = Input(shape=INPUT_SHAPE)

        fusion = Model(input_fusion, [ decoder(encoder(input_fusion)), classifier_end(encoder(input_fusion)) ], name='fusion')
        fusion.compile(optimizer=OPTIMIZER, 
              loss={
                  'decoder': 'mse', 
                  'classifier_end': 'categorical_crossentropy'},
              loss_weights={
                  'decoder': BETA, 
                  'classifier_end': 1.},
              metrics={
                  'decoder': 'accuracy', 
                  'classifier_end': 'accuracy'})
        
        score = fusion.evaluate(x=x_train,
	            y={"decoder": x_train, "classifier_end": y_train},
	            verbose=0)
        print("Accuracy du décodeur dans fusion:"+str(score[3]))
        print("Accuracy du classifieur dans fusion:"+str(score[4]))


    else:
        print("/!\ Unknown model : type 'autoencoder', 'classifier' or 'fusion'")
        exit(0)
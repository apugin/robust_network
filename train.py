import keras
from keras.layers import Input
from keras.models import Model
from params import INPUT_SHAPE, OPTIMIZER, BETA

def training(x_train, y_train ,model, nb_epoch, training_batch_size):
    if model == 'autoencoder':
        encoder = keras.models.load_model("saved_models/encoder.h5")
        decoder = keras.models.load_model("saved_models/decoder.h5")

        input = Input(shape=INPUT_SHAPE)
        autoencoder = Model(input, decoder(encoder(input)), name='autoencoder')
        autoencoder.compile(loss="mse", optimizer=OPTIMIZER, metrics=['accuracy'])

        loss = autoencoder.fit(x_train, x_train, batch_size=training_batch_size, epochs=nb_epoch, validation_split=0.1, callbacks=[], verbose=1)

        encoder.save("saved_models/encoder.h5")
        decoder.save("saved_models/decoder.h5")

    elif model == 'classifier':
        classifier_beginning = keras.models.load_model("saved_models/classifier_beginning.h5")
        classifier_end = keras.models.load_model("saved_models/classifier_end.h5")

        input_classifier = Input(shape=INPUT_SHAPE)
        classifier = Model(input_classifier, classifier_end(classifier_beginning(input_classifier)), name='classifier')
        classifier.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

        loss = classifier.fit(x_train, y_train, batch_size=training_batch_size, epochs=nb_epoch, validation_split=0.1, callbacks=[], verbose=1)

        classifier_beginning.save("saved_models/classifier_beginning.h5")
        classifier_end.save("saved_models/classifier_end.h5")

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
        
        loss = fusion.fit(x=x_train,
	                    y={"decoder": x_train, "classifier_end": y_train},
                        batch_size=training_batch_size,
	                    validation_split=0.1,
	                    epochs=nb_epoch,
	                    verbose=1)
        
        encoder.save("saved_models/encoder.h5")
        decoder.save("saved_models/decoder.h5")
        classifier_end.save("saved_models/classifier_end.h5")

    else:
        print("/!\ Unknown model : type 'autoencoder', 'classifier' or 'fusion'")
        exit(0)
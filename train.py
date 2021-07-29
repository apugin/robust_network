import keras
from keras.layers import Input
from keras.models import Model
from data import get_h_data, augmentation
from model import *
from params import INPUT_SHAPE, OPTIMIZER, BETA, K

def training(x_train, y_train, x_val, y_val , procedure, nb_epoch, training_batch_size,file):

    cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )

    if procedure: #If we use the procedure
        #Load and train the autoencoder
        encoder_path = "saved_models/encoder" + file + ".h5"
        decoder_path = "saved_models/decoder" + file + ".h5"

        encoder = load_model(encoder_path, "encoder", procedure)
        decoder = load_model(decoder_path, "decoder", procedure)

        autoencoder = assemble_autoencoder(encoder, decoder)

        loss = autoencoder.fit(x=x_train,
                       y=x_train, 
                       batch_size=training_batch_size, 
                       epochs=nb_epoch, 
                       validation_data=(x_val, x_val), 
                       shuffle=True, 
                       callbacks=[cb], 
                       verbose=1)

        encoder.save(encoder_path)
        decoder.save(decoder_path)
        
        #Create the databases h_train and h_val and augment h_train
        h_train, h_val = get_h_data(x_train, x_val, encoder)

        augmented_x, augmented_y = augmentation(h_train, y_train, K)

        #Load and train the classifier
        classifier_path = "saved_models/classifier_end" + file + ".h5"

        classifier = load_model(classifier_path, "classifier")

        #First, we train on non augmented database
        loss = classifier.fit(x=h_train,
	        y=y_train,
            batch_size=training_batch_size,
            validation_data=(h_val, y_val),
            callbacks=[cb],
	        epochs=nb_epoch,
	        verbose=1)
        
        #Then, we train on augmented database
        loss = classifier.fit(x=augmented_x,
	        y=augmented_y,
            batch_size=training_batch_size,
            validation_data=(h_val, y_val),
            callbacks=[cb],
	        epochs=nb_epoch,
	        verbose=1)
        
        classifier.save(classifier_path)

    else : #If we don't use the procedure
        #Load and train the "encoder" and "classifier" parts
        encoder_path = "saved_models/encoder" + file + ".h5"
        classifier_end_path = "saved_models/classifier_end" + file + ".h5"

        encoder = load_model(encoder_path, "encoder", procedure)
        classifier_end = load_model(classifier_end_path, "classifier_end", procedure)

        classifier = assemble_classifier(encoder, classifier_end)

        loss = classifier.fit(x=x_train,
	        y=y_train,
            batch_size=training_batch_size,
            validation_data=(x_val, y_val),
            callbacks=[cb],
	        epochs=nb_epoch,
	        verbose=1)
        
        encoder.save(encoder_path)
        classifier_end.save(classifier_end_path)


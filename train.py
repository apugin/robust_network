import keras
from keras.layers import Input
from keras.models import Model
from data import get_h_data, augmentation
from model import *
from params import INPUT_SHAPE, OPTIMIZER, BETA, K

def training(x_train, y_train ,model, nb_epoch, training_batch_size,file):
    cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=4, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )

    if model == 'autoencoder':
        encoder_path = "saved_models/encoder" + file + ".h5"
        decoder_path = "saved_models/decoder" + file + ".h5"

        encoder = load_model(encoder_path, "encoder")
        decoder = load_model(decoder_path, "decoder")

        autoencoder = assemble_autoencoder(encoder, decoder)

        loss = autoencoder.fit(x_train, x_train, batch_size=training_batch_size, epochs=nb_epoch, validation_split=0.1, callbacks=[cb], verbose=1)

        encoder.save(encoder_path)
        decoder.save(decoder_path)

    elif model == 'classifier':
        classifier_path = "saved_models/classifier_end" + file + ".h5"

        classifier = load_model(classifier_path, "classifier")

        h_train = get_h_data(x_train)

        augmented_h, augmented_y = augmentation(h_train, y_train, K)

        loss = classifier.fit(augmented_h, augmented_y, batch_size=training_batch_size, epochs=nb_epoch, validation_split=0.1, callbacks=[cb], verbose=1)

        classifier.save(classifier_path)

    elif model == 'fusion':
        encoder_path = "saved_models/encoder" + file + ".h5"
        decoder_path = "saved_models/decoder" + file + ".h5"
        classifier_end_path = "saved_models/classifier_end" + file + ".h5"

        encoder = load_model(encoder_path, "encoder")
        decoder = load_model(decoder_path, "decoder")
        classifier_end = load_model(classifier_end_path, "classifier_end")

        fusion = assemble_fusion(encoder, decoder, classifier_end)
        
        loss = fusion.fit(x=x_train,
	                    y={"decoder": x_train, "classifier_end": y_train},
                        batch_size=training_batch_size,
	                    validation_split=0.1,
	                    epochs=nb_epoch,
                        callbacks=[cb],
	                    verbose=1)
        
        encoder.save(encoder_path)
        decoder.save(decoder_path)
        classifier_end.save(classifier_end_path)

    else:
        print("/!\ Unknown model : type 'autoencoder', 'classifier' or 'fusion'")
        exit(0)

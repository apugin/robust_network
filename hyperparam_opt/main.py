import argparse
from data import get_data
import keras
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from model import AEHyperModel, sklearn_autoencoder, sklearn_classifier
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np

import warnings
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', dest='model', default='autoencoder', help="Choose the kind of model: 'autoencoder', 'classifier', 'fusion'")
parser.add_argument('--epoch', dest='epoch', type=int, default=40, help="Choose the number of epoch")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help="Choose the batch size")
parser.add_argument('--nb_samples', dest='nb_samples', type=int, default=5000, help="Choose the size of the training dataset")
parser.add_argument('--name', dest='name', default='', help="Choose the name of your search")

parser.add_argument('--search_type', dest='search_type', default='grid', help="Choose search type : 'random' or 'grid'(for autoencoder only)")
parser.add_argument('--max_trials', dest='max_trials', type=int, default=50, help="Choose the number of random trials")
parser.add_argument('--exec_per_trial', dest='exec_per_trial', type=int, default=2, help="Choose the number of executions per trial")
parser.add_argument('--seed', dest='seed', type=int, default=1, help="Choose the seed for random events")

args = parser.parse_args()


def main():

    (x_train, y_train), (x_test, y_test) = get_data(nb_training_samples=args.nb_samples)


    if args.name=='':
        filename = args.model + '_' + str(args.search_type) + '_samples_' + str(args.nb_samples) + '_seed_' + str(args.seed)
    else :
        filename = args.model + '_' + str(args.search_type) + '_samples_' + str(args.nb_samples) + '_seed_' + str(args.seed) + '_' + args.name

    if args.model=='autoencoder':
        if args.search_type=='random':
            hypermodel = AEHyperModel()
            tuner = RandomSearch(
                hypermodel,
                objective='val_loss',
                seed=args.seed,
                max_trials=args.max_trials,
                executions_per_trial=args.exec_per_trial,
                directory='random_search',
                project_name=filename
            )

            cb = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=4, verbose=0,
                mode='auto', baseline=None, restore_best_weights=True
            )

            tuner.search(x_train, x_train, 
                epochs=args.epoch, 
                batch_size=args.batch_size, 
                callbacks=[cb], 
                validation_split=0.1
            )

            tuner.results_summary()

            best_model = tuner.get_best_models(num_models=1)[0]

            loss = best_model.evaluate(x_test, x_test)
            print()
            print("Le meilleur modèle atteint sur la base de test une loss de :" + str(loss))

        if args.search_type=='grid':
            np.random.seed(args.seed)

            autoencoder =sklearn_autoencoder()

            batch_size = [args.batch_size]
            epochs = [args.epoch]
            nb_filters1 = [32, 64, 96]
            nb_filters2 = [16, 32, 64, 96]
            filter_size = [3, 4, 5]
            dim_latent = [10, 13, 16]
            param_grid = dict(batch_size=batch_size, epochs=epochs, nb_filters1=nb_filters1, nb_filters2=nb_filters2, dim_latent=dim_latent, filter_size=filter_size)
            grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid, n_jobs=1, verbose=3, cv=args.exec_per_trial)
            grid_result = grid.fit(x_train, x_train)

            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    elif args.model=='classifier':
        np.random.seed(args.seed)

        classifier = sklearn_classifier()

        batch_size = [args.batch_size]
        epochs = [args.epoch]
        nb_filters1 = [64]
        nb_filters2 = [96]
        filter_size = [5]
        dim_latent = [16]
        nb_neurons = [16,14,12]
        nb_layers = [0,1,2]

        param_grid = dict(batch_size=batch_size, epochs=epochs, nb_filters1=nb_filters1, nb_filters2=nb_filters2, dim_latent=dim_latent, filter_size=filter_size, nb_layers=nb_layers, nb_neurons=nb_neurons)
        grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=1, verbose=3, cv=args.exec_per_trial)


        grid_result = grid.fit(x_train, y_train)
    
    elif args.model=='fusion':

        kfold = KFold(n_splits=args.exec_per_trial, shuffle=True)
        puissance = np.linspace(-1.5, -0.5, 10)

        beta_test=[10**p for p in puissance]

        decoder_loss = []
        classifier_loss = []

        for beta in beta_test:
            decoder_loss_beta=0
            classifier_loss_beta=0
            for train, val in kfold.split(x_train, y_train):
                keras.backend.clear_session()
                fusion = load_fusion(beta)

                cb = keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=4, verbose=0,
                    mode='auto', baseline=None, restore_best_weights=True)

                loss = fusion.fit(x=x_train[train],
	                y={"decoder": x_train[train], "classifier_end": y_train[train]},
                    batch_size=128,
                    validation_data=(x_train[val], {'decoder': x_train[val], 'classifier_end': y_train[val]}),
                    callbacks=[cb],
	                epochs=50,
	                verbose=1)
                decoder_loss_beta += loss.history['decoder_loss'][-1]
                classifier_loss_beta += loss.history['classifier_end_loss'][-1]
  
            decoder_loss.append(decoder_loss_beta/nb_split)
            classifier_loss.append(classifier_loss_beta/nb_split)


        decoder_best = [0.010679 for i in range(len(beta_test))]
        classifier_best = [0.002991 for i in range(len(beta_test))]

        plt.figure()
        plt.plot(beta_test[0:], decoder_loss[0:])
        plt.plot(beta_test[0:], classifier_loss[0:])
        plt.plot(beta_test[0:], decoder_best[0:])
        plt.plot(beta_test[0:], classifier_best[0:])
        plt.legend(['Decoder loss', 'Classifier loss', 'Best decoder loss', 'Best classifier loss'])
        plt.ylabel('Loss value')
        plt.xlabel('Beta')
        plt.xscale('log')
        plt.show()

    else:
        print("/!\ Unknown model : type 'autoencoder', 'classifier' or 'fusion'")
        exit(0)

if __name__ == '__main__':
    main()

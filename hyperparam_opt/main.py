import argparse
from data import get_data
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from model import AEHyperModel

import warnings
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', dest='model', default='autoencoder', help="Choose the kind of model: 'autoencoder', 'classifier', 'fusion'")
parser.add_argument('--epoch', dest='epoch', type=int, default=40, help="Choose the number of epoch")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help="Choose the batch size")
parser.add_argument('--nb_samples', dest='nb_samples', type=int, default=5000, help="Choose the size of the training dataset")
parser.add_argument('--name', dest='name', default='', help="Choose the name of your search")

parser.add_argument('--search_type', dest='search_type', default='random', help="Choose search type : 'random or 'grid'")
parser.add_argument('--max_trials', dest='max_trials', type=int, default=50, help="Choose the number of random trials")
parser.add_argument('--exec_per_trial', dest='exec_per_trial', type=int, default=1, help="Choose the number of executions per trial")
parser.add_argument('--seed', dest='seed', type=int, default=1, help="Choose the seed for random events")

args = parser.parse_args()


def main():

    (x_train, y_train), (x_test, y_test) = get_data(nb_training_samples=args.nb_samples)


    if args.name=='':
        filename = args.model + '_' + str(args.search_type) + '_samples_' + str(args.nb_samples) + '_seed_' + str(args.seed)
    else :
        filename = args.model + '_' + str(args.search_type) + '_samples_' + str(args.nb_samples) + '_seed_' + str(args.seed) + '_' + args.name

    if args.model=='autoencoder':
        hypermodel = AEHyperModel()
    

    if args.search_type=='random':
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
        monitor='val_loss', min_delta=1e-3, patience=4, verbose=0,
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

        batch_size = [128]
        epochs = [50]
        nb_filters1 = [32, 64, 96]
        nb_filters2 = [16, 32, 64, 96]
        filter_size = [3, 4, 5]
        dim_latent = [10, 13, 16]
        param_grid = dict(batch_size=batch_size, epochs=epochs, nb_filters1=nb_filters1, nb_filters2=nb_filters2, dim_latent=dim_latent, filter_size=filter_size)
        grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid, n_jobs=1, verbose=3, cv=3)
        grid_result = grid.fit(x_train, x_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
    main()

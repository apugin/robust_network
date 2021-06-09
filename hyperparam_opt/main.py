import argparse
from data import get_data
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from model import AEHyperModel
from params import INPUT_SHAPE

import warnings
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', dest='model', default='autoencoder', help="Choose the kind of model: 'autoencoder', 'classifier', 'fusion'")
parser.add_argument('--epoch', dest='epoch', type=int, default=40, help="Choose the number of epoch")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help="Choose the batch size")
parser.add_argument('--nb_samples', dest='nb_samples', type=int, default=5000, help="Choose the size of the training dataset")
parser.add_argument('--name', dest='name', default='', help="Choose the name of your search")

parser.add_argument('--max_trials', dest='max_trials', type=int, default=50, help="Choose the number of random trials")
parser.add_argument('--exec_per_trial', dest='exec_per_trial', type=int, default=1, help="Choose the number of executions per trial")
parser.add_argument('--seed', dest='seed', type=int, default=1, help="Choose the seed for random events")

args = parser.parse_args()


def main():

    (x_train, y_train), (x_test, y_test) = get_data(nb_training_samples=args.nb_samples)

    if args.name=='':
        filename = args.model
    else :
        filename = args.model + '_' + args.name

    if args.model=='autoencoder':
        hypermodel = AEHyperModel(input_shape=INPUT_SHAPE)
    
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

if __name__ == '__main__':
    main()

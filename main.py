import argparse
import os
from data import get_data
from test import testing
from train import training

import warnings
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='')

parser.add_argument('--phase', dest='phase', default='test', help="Choose between 'train' or 'test'")
parser.add_argument('--model', dest='model', default='fusion', help="Choose the kind of model: 'autoencoder', 'classifier', 'fusion'")
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help="Choose the number of epoch")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help="Choose the batch size")
parser.add_argument('--nb_samples', dest='nb_samples', type=int, default=5000, help="Choose the size of the training dataset")

#parser.add_argument('--name', dest='name', default='', help="Choose the name of the model")

args = parser.parse_args()


def main():

    (x_train, y_train), (x_test, y_test) = get_data(nb_training_samples=args.nb_samples)

    if args.phase == 'train':
        training(x_train=x_train, y_train=y_train, model=args.model, nb_epoch=args.epoch, training_batch_size=args.batch_size)

    elif args.phase == 'test':
        testing(x_test=x_test, y_test=y_test, model=args.model)

    else:
        print("/!\ Unknown phase : type 'train' or 'test'")
        exit(0)


if __name__ == '__main__':
    main()

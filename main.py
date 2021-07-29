import argparse
from data import get_data
from test import testing
from train import training

import warnings
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='')

parser.add_argument('--phase', dest='phase', default='test', help="Choose between 'train' or 'test'")
parser.add_argument('--procedure', dest='procedure', type=bool, default=True, 
    help="For training: 'True' will train a model using the procedure and 'False' will train a model without the procedure")
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help="For training: Choose the number of epoch")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help="For training: Choose the batch size")
parser.add_argument('--nb_samples', dest='nb_samples', type=int, default=5000, help="For training: Choose the size of the training dataset")
parser.add_argument('--name', dest='name', default='new', help="For training: Choose the name of the model")

parser.add_argument('--epsilons', dest='epsilons', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], nargs='+', type=float,
    help="For testing: Choose list of epsilon to test robustness")

args = parser.parse_args()


def main():

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data(nb_training_samples=args.nb_samples)


    if args.phase == 'train':
        if args.name=='':
            filename = '_' + str(args.nb_samples)
            if args.procedure:
                filename += '_withprocedure'
            else:
                filename += '_noprocedure'
        else :
            filename = '_' + args.name
        training(x_train=x_train, 
            y_train=y_train, 
            x_val=x_val,
            y_val=y_val,
            procedure=args.procedure, 
            nb_epoch=args.epoch, 
            training_batch_size=args.batch_size, 
            file=filename)

    elif args.phase == 'test':
        testing(x_test=x_test, y_test=y_test, epsilons=args.epsilons)

    else:
        print("/!\ Unknown phase : type 'train' or 'test'")
        exit(0)


if __name__ == '__main__':
    main()

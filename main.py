"""main.py
Developer: Bayley King
Date: 2-19-2022
Descrition: Program controller
"""
################################## Imports ###################################
import argparse
import numpy as np
import pickle as pkl
import src.train as train
##############################################################################

################################# Constants ##################################
##############################################################################

def parse_arguments(args=None) -> list:
    """ Returns the parsed arguments.
        Parameters
        ----------
        args: List of strings to be parsed by argparse.
          The default None results in argparse using the values passed into sys.args.
    """
    parser = argparse.ArgumentParser(
            description="A program to run tensorflow networks.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args(args=args)
    return args

def main():
    a = np.load("data-set/x_train.npy")
    b = np.load("data-set/y_train.npy")
    c = np.load("data-set/x_test.npy")
    d = np.load("data-set/y_test.npy")
    train.train(a, b, c, d)


if __name__ == "__main__":
    import sys
    args = parse_arguments()
    try:
        main(**vars(args))
    except FileNotFoundError as exp:
        print(exp, file=sys.stderr)
        exit(-1)
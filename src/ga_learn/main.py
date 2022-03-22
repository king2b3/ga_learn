"""main.py
Developer: Bayley King
Date: 2-19-2022
Description: Program controller
"""
################################## Imports ###################################
import argparse
import numpy as np
import es as es
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
    parser = argparse.\
        ArgumentParser(description="A program to run tensorflow networks.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ng", "--number_of_generations",
                        help="Number of generations to run, barring "
                             "convergence.",
                        default=100, type=lambda gen: abs(int(gen)))
    args = parser.parse_args(args=args)
    return args


def main(number_of_generations: int = 100):
    train = {
        "x": np.load("data-set/x_train.npy"),
        "y": np.load("data-set/y_train.npy")
    }
    test = {
        "x": np.load("data-set/x_test.npy"),
        "y": np.load("data-set/y_test.npy")
    }
    algorithm = es.ES(train, test)
    algorithm.run(number_of_generations)


if __name__ == "__main__":
    import sys
    args = parse_arguments()
    try:
        main(**vars(args))
    except FileNotFoundError as exp:
        print(exp, file=sys.stderr)
        exit(-1)

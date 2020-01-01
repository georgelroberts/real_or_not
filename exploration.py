"""
Author: G. L. Roberts
Date: 01-01-2020
Description: A quick analysis of the train/test dataset.
Ideas:
    Hashtags should be more important features.
    Most entries have a 'keyword', but only ~2/3 have a location
"""

import os
import numpy as np
import pandas as pd

CDIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CDIR, 'data')


def main():
    explore_data()


def explore_data():
    train = load_dataset('train')
    test = load_dataset('test')
    print('Starting an exploratory data analysis.')
    print('Please see the README.md for more information\n')
    print('Train dataset information:')
    print(train.info())
    print()
    print('Test dataset information:')
    print(test.info())
    print()
    print('First 3 train rows:')
    print(train.head(3))
    print('First 3 test rows:')
    print(test.head(3))


def load_dataset(dset='train'):
    """
    Args:
        dset (str): train or test, default train
    Returns:
        pd.Dataframe: desired dataset
    """
    fname = os.path.join(DATA_DIR, f'{dset}.csv')
    return pd.read_csv(fname)


if __name__ == '__main__':
    main()

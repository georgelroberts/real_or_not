"""
Author: G. L. Roberts
Date: 01-01-2020
Description: A quick analysis of the train/test dataset.
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
    print('Train dataset information:')
    print(train.info())
    print('Test dataset information:')
    print(test.info())


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

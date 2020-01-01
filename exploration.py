"""
Author: G. L. Roberts
Date: 01-01-2020
Description: A quick analysis of the train/test dataset.
Ideas:
    Hashtags should be more important features.
    Most entries have a 'keyword', but only ~2/3 have a location
    Show length of tweets, any correlation with classification
    Number of hashtags as a feature
    Caps locks/punctuation could be important.
    Using location will almost certainly lead to overfitting.
"""

import os
import numpy as np
import pandas as pd
from extract_data import Extract_Data


def main():
    explore_data()


def explore_data():
    data_extractor = Extract_Data()
    train, test = data_extractor.load_data()
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


if __name__ == '__main__':
    main()

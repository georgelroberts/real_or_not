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
    explore = Exploration()
    explore()

class Exploration(object):
    def __init__(self):
        data_extractor = Extract_Data()
        self.train, self.test = data_extractor.load_data()

    def __call__(self):
        self.print_dataset_stats()

    def print_dataset_stats(self):
        print('Starting an exploratory data analysis.')
        print('Please see the README.md for more information\n')
        print('Train dataset information:')
        print(self.train.info())
        print()
        print('Test dataset information:')
        print(self.test.info())
        print()
        print('First 3 train rows:')
        print(self.train.head(3))
        print('First 3 test rows:')
        print(self.test.head(3))


if __name__ == '__main__':
    main()

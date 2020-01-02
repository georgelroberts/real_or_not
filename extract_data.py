"""
Author: G. L. Roberts
Date: 01-01-2020
Description: A class to extract/clean datasets
"""

import os
import pandas as pd

class Extract_Data(object):
    def __init__(self):
        self.cdir = os.path.abspath(os.path.dirname(__file__))
        self.data_dir = os.path.join(self.cdir, 'data')
        self.train_fname = os.path.join(self.data_dir, 'train.csv')
        self.test_fname = os.path.join(self.data_dir, 'test.csv')
    
    def load_data(self):
        """ Return pd DataFrame of train and test """
        self.train = pd.read_csv(self.train_fname)
        self.test = pd.read_csv(self.test_fname)
        return self.train, self.test
    
    def clean_data(self):
        pass

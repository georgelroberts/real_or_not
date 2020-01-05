"""
Author: G. L. Roberts
Date: 01-01-2020
Description: A class to extract/clean datasets.
Usage: Initialise class, then run get_train_test() to load datasets
        and extract features and target (test doesn't have target)
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


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
    
    def get_train_test(self):
        """ Load data and extract features, returning train X and y,
        alongside test X """
        self.load_data()
        train_X, test_X = self.extract_feats()
        train_y = self.train['target'].values
        return train_X, train_y, test_X
    
    def extract_feats(self):
        """ Return the X variables for train and test """
        train_text_df = self.train['text']
        text_clf = self.extraction_pipeline()
        train_X = text_clf.fit_transform(train_text_df.values)
        test_text_df = self.test['text']
        test_X = text_clf.transform(test_text_df.values)
        return train_X, test_X

    @staticmethod
    def extraction_pipeline():
        """ Method used to extract features """
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tdidf', TfidfTransformer())
        ])
        return text_clf

    def clean_data(self):
        pass

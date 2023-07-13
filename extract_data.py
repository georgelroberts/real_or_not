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
from textblob import TextBlob
from scipy.sparse import hstack

MAX_NB_WORDS = 50


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
    
    def get_train_test(self, clean=True):
        """ Load data and extract features, returning train X and y,
        alongside test X """
        self.load_data()
        self.add_to_train = self.more_feats('train')
        self.add_to_test = self.more_feats('test')
        if clean:
            self.clean_data()
        sub_df = self.test[['id']]
        train_X, test_X = self.extract_feats()
        train_y = self.train['target'].values
        return train_X, train_y, test_X, sub_df
    
    def extract_feats(self):
        """ Return the X variables for train and test """
        train_text_df = self.train['text']
        text_clf = self.extraction_pipeline()
        train_X = text_clf.fit_transform(train_text_df.values)
        # Use https://stackoverflow.com/a/41948136 to add columns to sparse arr
        train_X = hstack((train_X, self.add_to_train))
        test_text_df = self.test['text']
        test_X = text_clf.transform(test_text_df.values)
        test_X = hstack((test_X, self.add_to_test))
        return train_X, test_X

    @staticmethod
    def extraction_pipeline():
        """ Method used to extract features """
        return Pipeline(
            [('vect', CountVectorizer()), ('tdidf', TfidfTransformer())]
        )

    def more_feats(self, dset):
        if dset == 'train':
            data = self.train['text']
        elif dset == 'test':
            data = self.test['text'] 
        no_caps = data.str.findall(r'[A-Z]').str.len()
        tweet_length = data.str.len()
        no_hashtags = data.str.count(r'(\#\w+)')
        combined = pd.concat([tweet_length, no_caps, no_hashtags], axis=1)
        combined.columns = ['tweet_length', 'no_caps', 'no_hashtags']
        return combined

    def clean_data(self):
        self.make_lowercase()
        # self.correct_spelling()
        return self.train, self.test

    def make_lowercase(self):
        self.train['text'] = self.train['text'].str.lower()
        self.test['text'] = self.test['text'].str.lower()
    
    def correct_spelling(self):
        """ This function takes quite a long time..."""
        print("Correcting spelling")
        # Please see https://stackoverflow.com/a/35070548 for spellchecking
        self.train['text'].apply(lambda txt: ''.join(TextBlob(txt).correct()))
        self.test['text'].apply(lambda txt: ''.join(TextBlob(txt).correct()))
        print("Spelling correction done!")

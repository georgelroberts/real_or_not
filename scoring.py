"""
Author: G. L. Roberts
Date: 04-01-2020
About: Class to add scores and to view scores to a dataframe
        The dataframe has the columns 'model_name', 'cv_f1' and 'train_f1'
Usage: Initialise object, then use add_scores() to insert new scores, or
        print_all_scores() to view all scores. Alternatively, use the
        self.score_df variable directly.
"""

import os
import pickle
import pandas as pd
CDIR = os.path.abspath(os.path.dirname(__file__))

class Scoring(object):
    def __init__(self, df_fname='scores.pkl'):
        self.df_fname = df_fname
        self.results_fpath = os.path.join(CDIR, 'results', df_fname)
        self.load_df()

    def load_df(self):
        """ Load pickled dataframe on disk if it exists, else make it """
        if not os.path.exists(self.results_fpath):
            self.make_blank_df()
        else:
            with open(self.results_fpath, 'rb') as f:
                self.score_df = pickle.load(f)
    
    def make_blank_df(self):
        """ Initialise a blank dataframe and save it on disk """
        self.score_df = pd.DataFrame(columns=['model_name', 'cv_f1', 'train_f1'])
        self.save_df()

    def save_df(self):
        """ Save dataframe on disk """
        with open(self.results_fpath, 'wb') as f:
            pickle.dump(self.score_df, f)

    def add_scores(self, name, cv_score, train_score):
        """ Add scores to the dataframe and save it """
        is_in = self.check_if_in_df(name)
        if not is_in:
            position = len(self.score_df)
            self.score_df.loc[position] = [name, cv_score, train_score]
            self.save_df()

    def check_if_in_df(self, name):
        """ Check if model name already exists in the dataframe """
        if self.score_df['model_name'].str.contains(name).sum() > 0:
            print('Model already in dataframe')
            return True
        else:
            return False

    def print_best_score(self):
        """ Print only the best score """
        pass

    def print_all_scores(self):
        """ Print entire dataframe """
        print(self.score_df)

"""
Author: G. L. Roberts
Date: 04-01-2020
About: Class to add scores and to view scores to a dataframe
The dataframe has the columns 'model_name', 'cv_f1' and 'train_f1'
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
        if not os.path.exists(self.results_fpath):
            self.make_blank_df()
        else:
            with open(self.results_fpath, 'rb') as f:
                self.score_df = pickle.load(f)
    
    def make_blank_df(self):
        self.score_df = pd.DataFrame(columns=['model_name', 'cv_f1', 'train_f1'])
        self.save_df()

    def save_df(self):
        with open(self.results_fpath, 'wb') as f:
            pickle.dump(self.score_df, f)

    def add_scores(self, name, cv_score, train_score):
        position = len(self.score_df)
        self.score_df.loc[position] = [name, cv_score, train_score]

    def check_if_in_df(self):
        pass

    def print_best_score(self):
        pass

    def print_all_scores(self):
        print(self.score_df)

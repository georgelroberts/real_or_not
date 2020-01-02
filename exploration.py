"""
Author: G. L. Roberts
Date: 01-01-2020
Description: A quick analysis of the train/test dataset.
Ideas:
    Most entries have a 'keyword', but only ~2/3 have a location
    Number of hashtags as a feature, number of capital letters, punctuation,
        length of tweets
    Using location will almost certainly lead to overfitting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extract_data import Extract_Data
sns.set()


def main():
    explore = Exploration()
    explore()

class Exploration(object):
    def __init__(self):
        data_extractor = Extract_Data()
        self.train, self.test = data_extractor.load_data()

    def __call__(self):
        self.print_dataset_stats()
        self.number_of_hashtags_stats()
        self.length_of_tweets_stats()
        self.number_of_capital_letters_stats()

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

    def number_of_hashtags_stats(self):
        no_hashtags = self.train['text'].str.count(r'(\#\w+)')
        plotting_df = pd.DataFrame(columns=['no_hashtags', 'target'])
        plotting_df['no_hashtags'] = no_hashtags.values
        plotting_df['target'] = self.train['target'].values
        off_hist = plotting_df[plotting_df['target']==0]['no_hashtags'].values
        on_hist = plotting_df[plotting_df['target']==1]['no_hashtags'].values
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        max_no = np.max([off_hist.max(), on_hist.max()])
        ax1.hist(off_hist, range=(0, max_no), density=True)
        ax2.hist(on_hist, range=(0, max_no), density=True)
        ax1.set_title('Not a real disaster')
        ax2.set_title('Real disaster')
        ax1.set_xlabel('Number of hashtags')
        ax2.set_xlabel('Number of hashtags')
        plt.tight_layout()
        plt.show()

    def length_of_tweets_stats(self):
        tweet_length = self.train['text'].str.len()
        plotting_df = pd.DataFrame(columns=['tweet_length', 'target'])
        plotting_df['tweet_length'] = tweet_length.values
        plotting_df['target'] = self.train['target'].values
        off_hist = plotting_df[plotting_df['target']==0]['tweet_length'].values
        on_hist = plotting_df[plotting_df['target']==1]['tweet_length'].values
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        max_no = np.max([off_hist.max(), on_hist.max()])
        ax1.hist(off_hist, range=(0, max_no), density=True)
        ax2.hist(on_hist, range=(0, max_no), density=True)
        ax1.set_title('Not a real disaster')
        ax2.set_title('Real disaster')
        ax1.set_xlabel('Tweet length')
        ax2.set_xlabel('Tweet length')
        plt.tight_layout()
        plt.show()

    def number_of_capital_letters_stats(self):
        no_caps = self.train['text'].str.findall(r'[A-Z]').str.len()
        plotting_df = pd.DataFrame(columns=['no_caps', 'target'])
        plotting_df['no_caps'] = no_caps.values
        plotting_df['target'] = self.train['target'].values
        off_hist = plotting_df[plotting_df['target']==0]['no_caps'].values
        on_hist = plotting_df[plotting_df['target']==1]['no_caps'].values
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        max_no = np.max([off_hist.max(), on_hist.max()])
        ax1.hist(off_hist, range=(0, max_no), density=True)
        ax2.hist(on_hist, range=(0, max_no), density=True)
        ax1.set_title('Not a real disaster')
        ax2.set_title('Real disaster')
        ax1.set_xlabel('Number of caps')
        ax2.set_xlabel('Number of caps')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

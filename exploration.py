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
from collections import Counter
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
        # self.print_dataset_stats()
        # self.number_of_hashtags_stats()
        # self.length_of_tweets_stats()
        # self.number_of_capital_letters_stats()
        self.most_common_words_hist()

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
        off_hist, on_hist = self.extract_data_df(no_hashtags)
        self.hist_comparison_plotter(off_hist, on_hist, 'No. hashtags')

    def length_of_tweets_stats(self):
        tweet_length = self.train['text'].str.len()
        off_hist, on_hist = self.extract_data_df(tweet_length)
        self.hist_comparison_plotter(off_hist, on_hist, 'Tweet Length')

    def number_of_capital_letters_stats(self):
        no_caps = self.train['text'].str.findall(r'[A-Z]').str.len()
        off_hist, on_hist = self.extract_data_df(no_caps)
        self.hist_comparison_plotter(off_hist, on_hist, 'Number of capitals')
    
    def extract_data_df(self, data_df):
        plotting_df = pd.DataFrame(columns=['data', 'target'])
        plotting_df['data'] = data_df.values
        plotting_df['target'] = self.train['target'].values
        off_hist = plotting_df[plotting_df['target']==0]['data'].values
        on_hist = plotting_df[plotting_df['target']==1]['data'].values
        return off_hist, on_hist

    def most_common_words_hist(self, n_plot=30):
        on_texts = self.train[self.train['target']==1]['text'].str.split().values
        off_texts = self.train[self.train['target']==0]['text'].str.split().values
        on_texts_lst = [item for sublist in on_texts for item in sublist] 
        off_texts_lst = [item for sublist in off_texts for item in sublist] 
        on_texts_hst = Counter(on_texts_lst).most_common(n_plot)
        off_texts_hst = Counter(off_texts_lst).most_common(n_plot)

        _, (ax1, ax2) = plt.subplots(1, 2)
        on_df = pd.DataFrame(on_texts_hst, columns=['word', 'frequency'])
        on_df.plot(kind='barh', x='word', ax=ax1)
        off_df = pd.DataFrame(off_texts_hst, columns=['word', 'frequency'])
        off_df.plot(kind='barh', x='word', ax=ax2)
        plt.show()
    
    @staticmethod
    def hist_comparison_plotter(off_hist, on_hist, xlabel):
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        max_no = np.max([off_hist.max(), on_hist.max()])
        ax1.hist(off_hist, range=(0, max_no), density=True)
        ax2.hist(on_hist, range=(0, max_no), density=True)
        ax1.set_title('Not a real disaster')
        ax2.set_title('Real disaster')
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()

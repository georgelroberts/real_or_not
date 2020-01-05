"""
Author: G. L. Roberts
Date: 03-01-2020
About: Complete pipeline for extracting data, fitting and scoring
"""

import lightgbm as lgb
import os
from extract_data import Extract_Data
from scoring import Scoring
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

CDIR = os.path.abspath(os.path.dirname(__file__))


def main():
    Pipeline()()


class Pipeline(object):
    def __call__(self):
        """ Entire model pipeline, with data extraction, fitting and
        scoring """
        data_extractor = Extract_Data()
        train_X, train_y, test_X, sub_df = data_extractor.get_train_test()
        # clf = lgb.LGBMClassifier()
        clf = self.model() 
        name = 'MultinomialNB_default'
        self.fit_predict_and_score(train_X, train_y, name, clf)
        self.test_submission(clf, test_X, sub_df, name)

    def model(self):
        """ Returns the classifier """
        clf = MultinomialNB()
        return clf

    def fit_predict_and_score(self, train_X, train_y, name, clf):
        train_X, train_y, cv_X, cv_y = self.split_train_cv(train_X, train_y)
        clf.fit(train_X, train_y)
        train_pred = clf.predict(train_X)
        train_f1 = f1_score(train_y, train_pred)
        print(f'Train F1 score: {train_f1:.4f}')
        cv_pred = clf.predict(cv_X)
        cv_f1 = f1_score(cv_y, cv_pred)
        print(f'CV F1 score: {cv_f1:.4f}')

        scorer = Scoring()
        scorer.add_scores(name, cv_f1, train_f1)
        scorer.print_all_scores()
        return clf

    def split_train_cv(self, train_X, train_y):
        """ Split train into train and cross validation """
        train_X, cv_X, train_y, cv_y = train_test_split(
            train_X, train_y, test_size=0.33, random_state=42)
        return train_X, train_y, cv_X, cv_y

    def test_submission(self, clf, test_X, sub_df, fname):
        test_preds = clf.predict(test_X)
        sub_df['target'] = test_preds
        results_fname = os.path.join(CDIR, 'submissions', f'{fname}.csv')
        sub_df.to_csv(results_fname, index=False)


if __name__ == '__main__':
    main()
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
    pipeline()


def pipeline():
    """ Entire model pipeline, with data extraction, fitting and
    scoring """
    data_extractor = Extract_Data()
    train_X, train_y, test_X, sub_df = data_extractor.get_train_test()
    train_X, train_y, cv_X, cv_y = split_train_cv(train_X, train_y)
    # clf = lgb.LGBMClassifier()
    clf = MultinomialNB()
    clf.fit(train_X, train_y)
    train_pred = clf.predict(train_X)
    train_f1 = f1_score(train_y, train_pred)
    print(f'Train F1 score: {train_f1:.4f}')
    cv_pred = clf.predict(cv_X)
    cv_f1 = f1_score(cv_y, cv_pred)
    print(f'CV F1 score: {cv_f1:.4f}')

    scorer = Scoring()
    name = 'MultinomialNB_default'
    scorer.add_scores(name, cv_f1, train_f1)
    scorer.print_all_scores()

    test_submission(clf, test_X, sub_df, name)


def split_train_cv(train_X, train_y):
    """ Split train into train and cross validation """
    train_X, cv_X, train_y, cv_y = train_test_split(
        train_X, train_y, test_size=0.33, random_state=42)
    return train_X, train_y, cv_X, cv_y
    

def test_submission(clf, test_X, sub_df, fname):
    test_preds = clf.predict(test_X)
    sub_df['target'] = test_preds
    results_fname = os.path.join(CDIR, 'results', f'{fname}.csv')
    sub_df.to_csv(results_fname, index=False)


if __name__ == '__main__':
    main()
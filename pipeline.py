"""
Author: G. L. Roberts
Date: 03-01-2020
About: Complete pipeline for extracting data, fitting and scoring
"""

from extract_data import Extract_Data
from scoring import Scoring
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

def main():
    pipeline()


def pipeline():
    data_extractor = Extract_Data()
    train_X, train_y, test_X = data_extractor.get_train_test()
    train_X, train_y, cv_X, cv_y = split_train_cv(train_X, train_y)
    clf = MultinomialNB()
    clf.fit(train_X, train_y)
    train_pred = clf.predict(train_X)
    train_f1 = f1_score(train_y, train_pred)
    print(f'Train F1 score: {train_f1:.4f}')
    cv_pred = clf.predict(cv_X)
    cv_f1 = f1_score(cv_y, cv_pred)
    print(f'CV F1 score: {cv_f1:.4f}')

    scorer = Scoring()
    scorer.add_scores('MultinomialNB_default', cv_f1, train_f1)
    scorer.print_all_scores()
    return


def split_train_cv(train_X, train_y):
    train_X, cv_X, train_y, cv_y = train_test_split(
        train_X, train_y, test_size=0.33, random_state=42)
    return train_X, train_y, cv_X, cv_y
    

if __name__ == '__main__':
    main()
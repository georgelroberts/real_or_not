"""
Author: G. L. Roberts
Date: 03-01-2020
About: Complete pipeline for extracting data, fitting and scoring
"""

from extract_data import Extract_Data
from sklearn.model_selection import train_test_split

def main():
    data_extractor = Extract_Data()
    train_X, train_y, test_X = data_extractor.get_train_test()
    train_X, train_y, cv_X, cv_y = split_train_cv(train_X, train_y)
    breakpoint()


def split_train_cv(train_X, train_y):
    train_X, cv_X, train_y, cv_y = train_test_split(
        train_X, train_y, test_size=0.33, random_state=42)
    return train_X, train_y, cv_X, cv_y
    


if __name__ == '__main__':
    main()
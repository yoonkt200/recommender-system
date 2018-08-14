import argparse

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda, merge
from keras.optimizers import Adam


# pre-define about column info
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
          "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]
NUMERICAL_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


class WideAndDeep:
    def __init__(self):
        pass

class Wide:
    def __init__(self):
        pass

class Deep:
    def __init__(self):
        pass


def main():
    # prepare dataset
    df_train = pd.read_csv('adult.data.csv', header=None, names=COLUMNS)
    df_test = pd.read_csv('adult.test.csv', header=None, names=COLUMNS)

    df_train[LABEL_COLUMN] = df_train['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    df_test[LABEL_COLUMN] = df_test['income_bracket'].apply(lambda x: ">50K" in x).astype(int)

    y_train = df_train[LABEL_COLUMN].values
    y_test = df_test[LABEL_COLUMN].values

    df_train.pop(LABEL_COLUMN)
    df_test.pop(LABEL_COLUMN)

    x_train = df_train.values
    x_test = df_test.values

    # prepare hyper parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for networks')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epochs for the networks')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    args = parser.parse_args()
    model = WideAndDeep(args)
    model.fit()


if __name__ == '__main__':
    main()
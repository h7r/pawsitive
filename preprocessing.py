# -*- coding: utf-8 -*-
#
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path.cwd().parent / "data"


def prepare_data():
    # load the data
    df = pd.read_csv(DATA_DIR / "train.csv", index_col="id", header=0)
    df = df[["lines_per_sec", "distance", "pet_name"]]

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # shuffle the dataset
    df = df.sample(frac=1., random_state=33)

    # create dummies
    df = pd.get_dummies(df, drop_first=True)

    # change data types
    df = df.astype(np.float64)

    # split train/test
    h = int(df.shape[0] * 0.7)
    X_train, X_test = df.iloc[:h, 1:], df.iloc[h:, 1:]
    y_train, y_test = df.iloc[:h, 1], df.iloc[h:, 1]

    return X_train, X_test, y_train, y_test

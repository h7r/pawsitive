# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

MODELS_DIR = Path.cwd().parent / "models"
DATA_DIR = Path.cwd().parent / "data"
SUBMISSION_DIR = Path.cwd().parent / "data" / "submissions"


def training(X_train, X_test, y_train, y_test):
    # store the data in DDMatrix
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)

    # set params
    params = dict(max_depth=5, eta=0.1, tree_method="exact", objective="reg:squarederror", eval_metric="rmse",
                  nthread=4, seed=1302, gamma=0.04, interaction_constraints="[[0, 1], [0, 2], [0, 3]]", reg_lambda=0.4,
                  reg_alpha=0.1)

    # start training monitoring scores
    watchlist = [(d_test, "eval"), (d_train, "train")]
    bst = xgb.train(params, d_train, 15, watchlist)

    # boost from predictions
    ptrain = bst.predict(d_train, output_margin=True)
    ptest = bst.predict(d_test, output_margin=True)
    d_train.set_base_margin(ptrain)
    d_test.set_base_margin(ptest)
    best_booster = xgb.train(params, d_train, 15, watchlist)

    # save the model
    best_booster.save_model(MODELS_DIR / '003.model')

    return best_booster


def get_preds(model):
    # load and prepare testing set
    df = pd.read_csv(DATA_DIR / "test.csv", index_col="id", header=0)
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(np.float64)
    d_test = xgb.DMatrix(df)
    predictions = model.predict(d_test, iteration_range=(0, model.best_iteration))

    return predictions


def make_submission(predictions):
    df = pd.read_csv(DATA_DIR / "submission_format.csv", index_col="id", header=0)
    df["lines_per_sec"] = predictions
    df.to_csv(SUBMISSION_DIR / "submission003.csv")
    print(df.head(10))

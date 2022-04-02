# -*- coding: utf-8 -*-
#
from preprocessing import prepare_data
from visualization import create_figure
from train import *

if __name__ == "__main__":
    create_figure()
    X_train, X_test, y_train, y_test = prepare_data()
    bst = training(X_train, X_test, y_train, y_test)
    preds = get_preds(bst)
    make_submission(preds)

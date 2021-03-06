# -*- coding: utf-8 -*-
#

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR = Path.cwd().parent / "data"


def create_figure():
    df = pd.read_csv(DATA_DIR / "train.csv", index_col="id", header=0)
    sns.scatterplot(data=df, x="distance", y="lines_per_sec", hue="pet_name")
    plt.savefig("happy_fool_day.png")

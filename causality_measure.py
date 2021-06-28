# basic imports
import numpy as np
import pandas as pd

from scipy import stats
import matplotlib.pyplot as plt

# for ML
from sklearn.metrics import mutual_info_score
from scipy.ndimage.interpolation import shift

# os library
from os.path import exists, join


data_path = join(".", "data") if exists(join(".", "data")) \
    else join("..", "input", "statistical-learning-sapienza-spring-2021")


train = pd.read_csv(join(data_path, 'train.csv'), nrows=40)


len(train.id.values)
len(train.index.values)


def get_data(entry):
    id = entry.id
    variables = entry[['var1', 'var2', 'var3']].to_numpy()
    timeseries = entry[5 if "y" in entry.index else 4:].to_numpy(dtype=float).reshape((115, 116), order="F")
    if "y" in entry.index: return id, entry.y, variables, timeseries
    else: return id, variables, timeseries


timeseries = train.apply(lambda row: get_data(row)[-1], axis=1).to_list()

data = timeseries[0]

def lagged_timeseries(timeseries, steps):
    return np.array(list(map(lambda i: shift(timeseries, i, cval=np.NaN), range(steps))))

def get_symbols(timeseries):
    change = timeseries - shift(timeseries, 1, cval=0)
    for()
    return

manifolds = list(map(lambda x: lagged_timeseries(x, 3), timeseries))

symbol =

print(lagged_timeseries(data[:, 0], 4))

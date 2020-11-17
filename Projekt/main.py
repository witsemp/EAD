import pandas as pd
import pandasgui
import numpy as np
from matplotlib import pyplot as plt
import glob
import os


def task1():
    base_path = 'test/'
    filenames = [os.path.join(base_path, entry) for entry in os.listdir(base_path)]
    column_names = ['Name', 'Sex', 'Count']
    dataframes = [pd.read_csv(filename, delimiter=',', index_col=None, names=column_names) for filename in filenames]
    years = [year for year in range(1880, 2020)]
    for i, df in enumerate(dataframes):
        df['Year'] = years[i]
    df = pd.concat(dataframes)
    pandasgui.show(df, settings={'block': True})


if __name__ == '__main__':
    task1()

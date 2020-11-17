import pandas as pd
import pandasgui
import numpy as np
from matplotlib import pyplot as plt
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
    return df

def task2(df: pd.DataFrame):
    print("Number of unique names, regardless of sex: ", df['Name'].nunique())

def task3(df: pd.DataFrame):
    unique_male = df.loc[df['Sex'] == 'M', 'Name'].agg(['nunique']).iloc[0]
    unique_female = df.loc[df['Sex'] == 'F', 'Name'].agg(['nunique']).iloc[0]
    male_names = df.loc[df['Sex'] == 'M', ['Name', 'Year']]
    female_names = df.loc[df['Sex'] == 'F', ['Name', 'Year']]
    pandasgui.show(male_names, settings={'block': True})
    pandasgui.show(female_names, settings={'block': True})
    print("Number of unique female names: ", unique_female)
    print("Number of unique male names: ", unique_male)
if __name__ == '__main__':
    df = task1()
    task2(df)
    task3(df)

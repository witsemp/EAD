import pandas as pd
import pandasgui
import numpy as np
from matplotlib import pyplot as plt
import os

years = [year for year in range(1880, 2020)]


def task1():
    base_path = 'names/'
    filenames = [os.path.join(base_path, entry) for entry in os.listdir(base_path)]
    column_names = ['Name', 'Sex', 'Count']
    dataframes = [pd.read_csv(filename, delimiter=',', index_col=None, names=column_names) for filename in filenames]
    for i, df in enumerate(dataframes):
        df['Year'] = years[i]
    df = pd.concat(dataframes)
    df.index = np.arange(start=0, stop=len(df.index))
    return df


def task2(df: pd.DataFrame):
    print("Number of unique names, regardless of sex: ", df['Name'].nunique())


def task3(df: pd.DataFrame):
    unique_male = df.loc[df['Sex'] == 'M', 'Name'].agg(['nunique']).iloc[0]
    unique_female = df.loc[df['Sex'] == 'F', 'Name'].agg(['nunique']).iloc[0]
    print("Number of unique female names: ", unique_female)
    print("Number of unique male names: ", unique_male)


def task4(df: pd.DataFrame):
    # print(df.pivot_table(index="Year", columns=["Name", "Sex"], aggfunc='sum'))
    df['frequency_male'] = 0
    df['frequency_female'] = 0

    for year in years:
        total_yearly_male_births = df.loc[(df['Sex'] == 'M') & (df['Year'] == year), 'Count'].sum()
        total_yearly_female_births = df.loc[(df['Sex'] == 'F') & (df['Year'] == year), 'Count'].sum()
        df.loc[(df['Sex'] == 'M') & (df['Year'] == year), 'frequency_male'] = df['Count']/total_yearly_male_births
        df.loc[(df['Sex'] == 'F') & (df['Year'] == year), 'frequency_female'] = df['Count'] / total_yearly_female_births
    print(df)


if __name__ == '__main__':
    df = task1()
    # task2(df)
    # task3(df)
    task4(df)

import pandas as pd
import pandasgui
import numpy as np
from matplotlib import pyplot as plt
import os

# pandasgui.show(df, settings={'block': True})
years = [year for year in range(1880, 2020)]


def task1():
    base_path = 'names/'
    filenames = [os.path.join(base_path, entry) for entry in os.listdir(base_path)]
    column_names = ['Name', 'Gender', 'Count']
    dataframes = [pd.read_csv(filename, delimiter=',', index_col=None, names=column_names) for filename in filenames]
    for i, df in enumerate(dataframes):
        df['Year'] = years[i]
    df = pd.concat(dataframes)
    df.index = np.arange(start=0, stop=len(df.index))
    return df


def task2(df: pd.DataFrame):
    print("Number of unique names, regardless of Gender: ", df['Name'].nunique())


def task3(df: pd.DataFrame):
    unique_male = df.loc[df['Gender'] == 'M', 'Name'].agg(['nunique']).iloc[0]
    unique_female = df.loc[df['Gender'] == 'F', 'Name'].agg(['nunique']).iloc[0]
    print("Number of unique female names: ", unique_female)
    print("Number of unique male names: ", unique_male)


def task4(df: pd.DataFrame):
    df['frequency_male'] = 0
    df['frequency_female'] = 0
    for year in years:
        total_yearly_male_births = df.loc[(df['Gender'] == 'M') & (df['Year'] == year), 'Count'].sum()
        total_yearly_female_births = df.loc[(df['Gender'] == 'F') & (df['Year'] == year), 'Count'].sum()
        df.loc[(df['Gender'] == 'M') & (df['Year'] == year), 'frequency_male'] = df['Count'] / total_yearly_male_births
        df.loc[(df['Gender'] == 'F') & (df['Year'] == year), 'frequency_female'] = df[
                                                                                       'Count'] / total_yearly_female_births

    return df


def task5(df: pd.DataFrame):
    years_as_rows_names = pd.pivot_table(df, index='Year', values='Count', columns=['Name'], aggfunc='sum')
    years_as_rows_gender = pd.pivot_table(df, index='Year', values='Count', columns=['Gender'], aggfunc='sum')
    years_as_rows_names['Births'] = years_as_rows_names.sum(axis=1)
    years_as_rows_names['Female births'] = years_as_rows_gender['F']
    years_as_rows_names['Male births'] = years_as_rows_gender['M']
    years_as_rows_names['F/M'] = years_as_rows_names['Female births'] / years_as_rows_names['Male births']
    years_as_rows_names['Difference'] = np.abs(
        years_as_rows_names['Female births'] - years_as_rows_names['Male births'])
    print("Year with biggest difference: ", years_as_rows_names['Difference'].idxmax())
    print("Year with smallest difference: ", years_as_rows_names['Difference'].idxmin())
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(years_as_rows_names.index.values, years_as_rows_names['Births'].values, 'or')
    ax1.set_title('Total number of births per year')
    ax2.plot(years_as_rows_names.index.values, years_as_rows_names['F/M'].values, 'or')
    ax2.set_title('Female to male births ratio per year')
    fig.tight_layout()
    plt.show()


def task6(df: pd.DataFrame):

    male_df = pd.pivot_table(df.loc[df['Gender'] == 'M', :], index='Year', values='Count', columns=['Name'],
                             aggfunc='sum', fill_value=0.0)
    female_df = pd.pivot_table(df.loc[df['Gender'] == 'F', :], index='Year', values='Count', columns=['Name'],
                               aggfunc='sum', fill_value=0.0)
    print(male_df.sum(axis=0).sort_values(axis=0, ascending=False).head(1000))
    print(female_df.sum(axis=0).sort_values(axis=0, ascending=False).head(1000))




if __name__ == '__main__':
    df = task1()
    # task2(df)
    # task3(df)
    df_frequency = task4(df)
    task5(df_frequency)
    task6(df_frequency)

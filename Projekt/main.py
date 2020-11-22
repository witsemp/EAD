import pandas as pd
import pandasgui
import numpy as np
from matplotlib import pyplot as plt
import os
import time

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


# TODO: rebuild task 4
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
    return years_as_rows_names


def task6(df: pd.DataFrame):
    male_df = pd.pivot_table(df.loc[df['Gender'] == 'M', :], index='Year', values='Count', columns=['Name'],
                             aggfunc='sum', fill_value=0.0)
    m1000_names_per_year = male_df.sort_values(by=list(male_df.index.values), axis=1, ascending=False).iloc[:,
                           :1000].sum(axis=1)
    m1000_names_per_year = pd.DataFrame(m1000_names_per_year).rename(columns={0: "Top 1000 count"})
    female_df = pd.pivot_table(df.loc[df['Gender'] == 'F', :], index='Year', values='Count', columns=['Name'],
                               aggfunc='sum', fill_value=0.0)
    f1000_names_per_year = female_df.sort_values(by=list(female_df.index.values), axis=1, ascending=False).iloc[:,
                           :1000].sum(axis=1)
    f1000_names_per_year = pd.DataFrame(f1000_names_per_year).rename(columns={0: "Top 1000 count"})
    male_ranking = male_df.sum(axis=0).sort_values(axis=0, ascending=False).head(1000)
    female_ranking = female_df.sum(axis=0).sort_values(axis=0, ascending=False).head(1000)
    return male_ranking, female_ranking, m1000_names_per_year, f1000_names_per_year


def task7(df: pd.DataFrame, m_ranking: pd.DataFrame, f_ranking: pd.DataFrame):
    def _get_sum_for_name(name: str, frame: pd.DataFrame):
        return pd.pivot_table(frame.loc[df['Name'] == name, :], values='Count', index='Year', aggfunc='sum')

    def _get_frequency_for_name(name: str, frame: pd.DataFrame):
        return pd.pivot_table(frame.loc[df['Name'] == name, :], values=['frequency_male', 'frequency_female'],
                              index='Year', aggfunc='sum').sum(axis=1)

    def _format_ax(ax, name: str, row: int, col: int, frame_f: pd.DataFrame, frame_s: pd.DataFrame):
        ax[row][col].plot(frame_s.index.values, frame_s.values)
        ax[row][col].set_title(name)
        ax[row][col].set_ylabel('Total number change')
        secax = ax[row][col].twinx()
        secax.plot(frame_f.index.values, frame_f.values, color="red")
        secax.set_ylabel('Frequency change')

    top_male_name = m_ranking.head(1).index[0]
    print("Most common male name: ", top_male_name)
    top_female_name = f_ranking.head(1).index[0]
    print("Most common female name: ", top_female_name)
    harry_sum = _get_sum_for_name('Harry', df)
    harry_f = _get_frequency_for_name('Harry', df)
    marilyn_sum = _get_sum_for_name('Marilin', df)
    marilyn_f = _get_frequency_for_name('Marilin', df)
    top_male_sum = _get_sum_for_name(top_male_name, df)
    top_male_frequency = _get_frequency_for_name(top_male_name, df)
    top_female_sum = _get_sum_for_name(top_female_name, df)
    top_female_frequency = _get_frequency_for_name(top_female_name, df)
    fig, ax = plt.subplots(2, 2)
    _format_ax(ax, 'Harry', 0, 0, harry_f, harry_sum)
    _format_ax(ax, 'Marilin', 0, 1, marilyn_f, marilyn_sum)
    _format_ax(ax, top_male_name, 1, 0, top_male_frequency, top_male_sum)
    _format_ax(ax, top_female_name, 1, 1, top_female_frequency, top_female_sum)
    fig.tight_layout()
    plt.show()


def task8(frame: pd.DataFrame, m_ranking, f_ranking):
    frame['Top 1000 male ratio'] = m_ranking['Top 1000 count'] / frame['Male births']
    frame['Top 1000 female ratio'] = f_ranking['Top 1000 count'] / frame['Female births']
    fig, ax = plt.subplots()
    ax.plot(frame.index.values, frame['Top 1000 male ratio'].values, color='blue')
    ax.plot(frame.index.values, frame['Top 1000 female ratio'].values, color='red')
    ax.legend(['Top 1000 names male ratio', 'Top 1000 names female ratio'])
    ax.set_title("Top 1000 names to all names ratio by year")
    plt.show()


def task9(frame: pd.DataFrame, frame_statistics: pd.DataFrame):
    frame['Last letter'] = df['Name'].str.strip().str[-1]
    pivot_frame = frame.pivot_table(index=['Year'], columns=['Last letter', 'Gender'], values='Count', aggfunc='sum')
    pivot_frame = pivot_frame.div(pivot_frame.sum(axis=1), axis=0)
    all_years = pivot_frame.copy()
    pivot_frame = pivot_frame.loc[[1910, 1960, 2015], :]
    pivot_frame = pivot_frame.iloc[:, pivot_frame.columns.get_level_values(1) == 'M']
    pivot_frame.columns = pivot_frame.columns.droplevel(1)
    pivot_frame = pivot_frame.rename_axis(None, axis=1)
    ax = pivot_frame.T.plot(kind='bar')
    # TODO: Change xticks rotation
    plt.show()
    pivot_frame = pivot_frame.transpose()
    pivot_frame['Change over period'] = np.abs(pivot_frame[2015] - pivot_frame[1910])
    pivot_frame.sort_values(by='Change over period', axis=0, ascending=False, inplace=True)
    most_popular_letters = pivot_frame.head(3).index.values
    # print("Letter with biggest change between 2015 and 1910 is: ", most_popular_letters[0])
    all_years = all_years.iloc[:, all_years.columns.get_level_values(1) == 'M']
    all_years.columns = all_years.columns.droplevel(1)
    all_years = all_years.rename_axis(None, axis=1)
    all_years = all_years.transpose()
    all_years = all_years.loc[most_popular_letters, :]
    print(all_years)
    # TODO: Add description to plot
    fig, ax = plt.subplots()
    ax.plot(all_years.columns.values, all_years.loc['n', :], color='red')
    ax.plot(all_years.columns.values, all_years.loc['r', :], color='green')
    ax.plot(all_years.columns.values, all_years.loc['d', :], color='blue')
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    df = task1()
    # task2(df)
    # task3(df)
    df_frequency = task4(df)
    df_frequency_pivot = task5(df_frequency)
    male_ranking, female_ranking, m1000_per_year, f_1000_per_year = task6(df_frequency)
    # task7(df_frequency, male_ranking, female_ranking)
    # task8(df_frequency_pivot, m1000_per_year, f_1000_per_year)
    task9(df_frequency, df_frequency_pivot)
    print(time.time() - start_time)

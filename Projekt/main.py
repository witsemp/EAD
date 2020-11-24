import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sqlite3

years = [year for year in range(1880, 2020)]
years_mortality = [year for year in range(1959, 2018)]


def task1():
    base_path = 'names/'
    filenames = [os.path.join(base_path, entry) for entry in os.listdir(base_path)]
    column_names = ['Name', 'Gender', 'Count']
    dataframes = [pd.read_csv(filename, delimiter=',', index_col=None, names=column_names) for filename in filenames]
    for i, df in enumerate(dataframes):
        df['Year'] = years[i]
    df = pd.concat(dataframes)
    df.index = np.arange(start=0, stop=len(df.index))
    print("Zadanie 1: Tablica z danymi")
    print(df)
    return df


def task2(df: pd.DataFrame):
    print("Zadanie 2: Ilość unikalnych imion nadanych w tym czasie to:  ", df['Name'].nunique())


def task3(df: pd.DataFrame):
    unique_male = df.loc[df['Gender'] == 'M', 'Name'].agg(['nunique']).iloc[0]
    unique_female = df.loc[df['Gender'] == 'F', 'Name'].agg(['nunique']).iloc[0]
    print("Zadanie 3: Ilość unikalnych imion żeńskich nadanych w tym czasie to: ", unique_female)
    print("Zadanie 3: Ilość unikalnych imion męskich nadanych w tym czasie to: ", unique_male)


def task4(df: pd.DataFrame):
    df['frequency_male'] = 0
    df['frequency_female'] = 0
    df['frequency_male'] = df['frequency_male'].astype(np.float64)
    df['frequency_female'] = df['frequency_female'].astype(np.float64)
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
    print("Zadanie 5: Rok z największą różnicą: ", years_as_rows_names['Difference'].idxmax())
    print("Zadanie 5: Rok z najmniejszą różnicą: ", years_as_rows_names['Difference'].idxmin())
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(years_as_rows_names.index.values, years_as_rows_names['Births'].values, 'r')
    ax1.set_title('Całkowita roczna liczba urodzin')
    ax2.plot(years_as_rows_names.index.values, years_as_rows_names['F/M'].values, 'r')
    ax2.set_title('Stosunek liczby narodzin dziewczynek do liczby narodzin chłopców')
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
    print("Zadanie 6: Ranking imion męskich")
    print(pd.DataFrame(male_ranking))
    print("Zadanie 6: Ranking imion żeńskich")
    print(pd.DataFrame(female_ranking))
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
        ax[row][col].set_ylabel('Całkowita roczna liczba urodzeń')
        secax = ax[row][col].twinx()
        secax.plot(frame_f.index.values, frame_f.values, color="red")
        secax.set_ylabel('Roczna popularność')

    top_male_name = m_ranking.head(1).index[0]
    print("Zadanie 7: Najpopularniejsze imię męskie: ", top_male_name)
    top_female_name = f_ranking.head(1).index[0]
    print("Zadanie 7: Najpopularniejsze imię żeńskie: ", top_female_name)
    harry_sum = _get_sum_for_name('Harry', df)
    harry_f = _get_frequency_for_name('Harry', df)
    marilyn_sum = _get_sum_for_name('Marilin', df)
    marilyn_f = _get_frequency_for_name('Marilin', df)
    top_male_sum = _get_sum_for_name(top_male_name, df)
    top_male_frequency = _get_frequency_for_name(top_male_name, df)
    top_female_sum = _get_sum_for_name(top_female_name, df)
    top_female_frequency = _get_frequency_for_name(top_female_name, df)
    frame_names = df.pivot_table(index='Name', columns='Year', values='Count', aggfunc='sum')
    count_harry = frame_names.loc['Harry', [1940, 1980, 2019]].values
    count_marilin = frame_names.loc['Marilin', [1940, 1980, 2019]].values
    count_marilin[0] = 0
    count_top_male = frame_names.loc[top_male_name, [1940, 1980, 2019]].values
    count_top_female = frame_names.loc[top_female_name, [1940, 1980, 2019]].values
    print("Zadanie 7: Imię Harry w latach 1940, 1980, 2019 nadane odpowiednio ", count_harry[0], count_harry[1],
          count_harry[2], ' razy')
    print("Zadanie 7: Imię Marilin w latach 1940, 1980, 2019 nadane odpowiednio ", count_marilin[0], count_marilin[1],
          count_marilin[2], ' razy')
    print("Zadanie 7: Imię James w latach 1940, 1980, 2019 nadane odpowiednio ", count_top_male[0], count_top_male[1],
          count_top_male[2], ' razy')
    print("Zadanie 7: Imię Mary w latach 1940, 1980, 2019 nadane odpowiednio ", count_top_female[0],
          count_top_female[1],
          count_top_female[2], ' razy')
    fig, ax = plt.subplots(2, 2)
    _format_ax(ax, 'Harry', 0, 0, harry_f, harry_sum)
    _format_ax(ax, 'Marilin', 0, 1, marilyn_f, marilyn_sum)
    _format_ax(ax, top_male_name, 1, 0, top_male_frequency, top_male_sum)
    _format_ax(ax, top_female_name, 1, 1, top_female_frequency, top_female_sum)
    fig.tight_layout()
    plt.show()


def task8(frame: pd.DataFrame, m_ranking, f_ranking):
    frame['Top 1000 male ratio'] = (m_ranking['Top 1000 count'] / frame['Male births'])*100.0
    frame['Top 1000 female ratio'] = (f_ranking['Top 1000 count'] / frame['Female births'])*100.0
    frame['Top 1000 difference'] = np.abs(frame['Top 1000 male ratio'] - frame['Top 1000 female ratio'])
    max_difference_year = frame['Top 1000 difference'].idxmax()
    print("Zadanie 8: Rok, w którym zaobserwowano największą różnicę w różnorodności: ", max_difference_year)
    fig, ax = plt.subplots()
    ax.plot(frame.index.values, frame['Top 1000 male ratio'].values, color='blue')
    ax.plot(frame.index.values, frame['Top 1000 female ratio'].values, color='red')
    ax.legend(['Udział imion męskich', 'Udział imion żeńskich'])
    ax.set_title("Roczny procentowy udział imion z rankingu top 1000")
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
    ax1 = pivot_frame.T.plot(kind='bar')
    ax1.set_title("Popularność ostatniej litery imienia dla mężczyzn z podziałem na lata")
    plt.show()
    pivot_frame = pivot_frame.transpose()
    pivot_frame['Change over period'] = np.abs(pivot_frame[2015] - pivot_frame[1910])
    pivot_frame['Change with sign'] = pivot_frame[2015] - pivot_frame[1910]
    pivot_frame.sort_values(by='Change over period', axis=0, ascending=False, inplace=True)
    max_increase = pivot_frame['Change with sign'].idxmax()
    max_decrease = pivot_frame['Change with sign'].idxmin()
    print('Zadanie 9: Maksymalny przyrost wystąpił dla litery: ', max_increase)
    print('Zadanie 9: Maksymalny spadek wystąpił dla litery: ', max_decrease)
    most_popular_letters = pivot_frame.head(3).index.values
    print('Zadanie 9: Największy wzrost/spadek wystąpił dla litery: ', most_popular_letters[0])
    all_years = all_years.iloc[:, all_years.columns.get_level_values(1) == 'M']
    all_years.columns = all_years.columns.droplevel(1)
    all_years = all_years.rename_axis(None, axis=1)
    all_years = all_years.transpose()
    all_years = all_years.loc[most_popular_letters, :]
    fig, ax = plt.subplots()
    ax.plot(all_years.columns.values, all_years.loc['n', :], color='red')
    ax.plot(all_years.columns.values, all_years.loc['r', :], color='green')
    ax.plot(all_years.columns.values, all_years.loc['d', :], color='blue')
    ax.set_title("Przebieg trendu popularności w maksymalnym przedziale czasu")
    ax.legend(['n', 'r', 'd'])
    plt.show()


def task10(frame: pd.DataFrame):
    df_male = frame.loc[frame['Gender'] == 'M', :]
    df_female = frame.loc[frame['Gender'] == 'F', :]
    df_male = df_male.set_index(['Name', 'Year'])
    df_female = df_female.set_index(['Name', 'Year'])
    merged = df_male.merge(df_female, left_index=True, right_index=True)
    merged_grouped = merged.groupby(level='Name').sum()
    most_frequent_male = merged_grouped['Count_x'].idxmax()
    most_frequent_female = merged_grouped['Count_y'].idxmax()
    print("Zadanie 10: Najpopularniejsze imię męskie nadawane też dziewczynkom to: ", most_frequent_male)
    print("Zadanie 10: Najpopularniejsze imię żeńskie nadawane też chłopcom to: ", most_frequent_female)
    return merged


def task11(frame: pd.DataFrame):
    frame['Popularity'] = frame['frequency_male_x'] / (frame['frequency_male_x'] + frame['frequency_female_y'])
    frame = frame.rename_axis(['Name', 'Year'])
    frame_queried_1880 = frame.query("(Year >= 1880 & Year <= 1920)").loc[:, 'Popularity']
    frame_queried_2000 = frame.query("(Year >= 2000 & Year <= 2019)").loc[:, 'Popularity']
    frame_grouped_1880 = pd.DataFrame(frame_queried_1880.groupby(level='Name').mean())
    frame_grouped_1880.columns = ['Popularity']
    frame_grouped_2000 = pd.DataFrame(frame_queried_2000.groupby(level='Name').mean())
    frame_grouped_2000.columns = ['Popularity']
    merged = frame_grouped_1880.merge(frame_grouped_2000, left_index=True, right_index=True)
    merged.columns = ['Popularity_x', 'Popularity_y']
    merged['Popularity difference'] = np.abs(merged['Popularity_x'] - merged['Popularity_y'])
    merged.sort_values(by='Popularity difference', ascending=False, inplace=True)
    names = merged.head(2).index.values
    print("Zadanie 11: Najpopularniejsze imiona: ", names)
    frame_copy = frame.reset_index()
    frame_copy = frame_copy.loc[:, ['Year', 'Name', 'Count_x', 'Count_y']]
    frame_copy = frame_copy.set_index('Name')
    name1_df = frame_copy.loc[names[0], :]
    name2_df = frame_copy.loc[names[1], :]


def task12():
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")
    frame = pd.read_sql_query('SELECT * FROM USA_fltper_1x1 UNION ALL SELECT * FROM USA_mltper_1x1', conn)
    conn.close()
    return frame


def task13(frame1: pd.DataFrame, frame2: pd.DataFrame):
    frame1_queried = frame1.loc[:, ['Year', 'dx']]
    frame1_pivot = pd.DataFrame(frame1_queried).pivot_table(index='Year', values='dx', aggfunc='sum')
    frame2_queried = pd.DataFrame(frame2.loc[1959:2017, 'Births'])
    frame2_queried.columns = ['Births']
    frame2_queried['Difference'] = frame2_queried['Births'] - frame1_pivot['dx']
    rate_sum = frame2_queried['Difference'].sum()
    rate_mean = frame2_queried['Difference'].mean()
    print("Zadanie 13: Całkowity przyrost w okresie 1959-2017 wynosi: ", rate_sum)
    print("Zadanie 13: Średni przyrost w okresie 1959-2017 wynosi: ", rate_mean)
    return frame2_queried


def task14(frame1: pd.DataFrame, frame2: pd.DataFrame):
    frame_queried = frame1.query("Age == 0").loc[:, ['Year', 'dx']]
    frame_queried = pd.DataFrame(frame_queried).pivot_table(index='Year', values='dx', aggfunc='sum')
    frame2['Survival rate'] = (frame2['Births'] - frame_queried['dx']) / frame2['Births']
    fig, ax = plt.subplots()
    ax.set_title("Wartość współczynnika przeżywalności dzieci")
    ax.plot(frame2.index.values, frame2['Survival rate'].values)
    return ax


def task15(frame1: pd.DataFrame, frame2: pd.DataFrame, axis):
    def _get_5_year_period_mortality(year, df1: pd.DataFrame):
        mortality_sum = 0
        df_queried = pd.DataFrame(df1.loc[((df1['Year'] >= year) & (df1['Year'] <= year + 4)), :])
        for i in range(5):
            mortality_sum = mortality_sum + \
                            df_queried.loc[(df_queried['Year'] == year + i) & (df_queried['Age'] == i), 'dx'].sum()
        return mortality_sum

    fq = pd.DataFrame(frame1.loc[:, ['Year', 'Age', 'Sex', 'dx']])
    aggregated_mortality = [_get_5_year_period_mortality(year, fq) for year in years_mortality]
    survival_ratio = (frame2['Births'].values - aggregated_mortality) / frame2['Births'].values
    axis.plot(years_mortality, survival_ratio)
    axis.legend(['Przeżywalność w pierwszym roku życia', "Przeżywalność w pierwszych 5 latach życia"])
    plt.show()


if __name__ == '__main__':
    df = task1()
    task2(df)
    task3(df)
    df_frequency = task4(df)
    df_frequency_pivot = task5(df_frequency)
    male_ranking, female_ranking, m1000_per_year, f_1000_per_year = task6(df_frequency)
    task7(df_frequency, male_ranking, female_ranking)
    task8(df_frequency_pivot, m1000_per_year, f_1000_per_year)
    task9(df_frequency, df_frequency_pivot)
    df_fm_names = task10(df_frequency)
    task11(df_fm_names)
    sql_frame = task12()
    df_births_by_year = task13(sql_frame, df_frequency_pivot)
    axs = task14(sql_frame, df_births_by_year)
    task15(sql_frame, df_births_by_year, axs)

import os
import pandas as pd
import pandasgui
from typing import List
import datetime as dt


# pd.options.display.max_columns = None


def read_data():
    base_path = 'data'
    confirmed = pd.read_csv(os.path.join(base_path, 'time_series_covid19_confirmed_global.csv'))
    deaths = pd.read_csv(os.path.join(base_path, 'time_series_covid19_deaths_global.csv'))
    recovered = pd.read_csv(os.path.join(base_path, 'time_series_covid19_recovered_global.csv'))
    # pandasgui.show(deaths, settings={'block': True})
    return confirmed, deaths, recovered


def expand(frame: pd.DataFrame) -> pd.DataFrame:
    frame['Country with province'] = frame['Country/Region'] + ' ' + frame['Province/State'].fillna('').astype(str)
    # TODO: Possible problem with space
    frame.set_index('Country with province', inplace=True)
    return frame


def identify_countries_wo_recovered_info(confirmed: pd.DataFrame, recovered: pd.DataFrame) -> List:
    all_countries = confirmed.index.values
    countries_w_recovered_info = recovered.index.values
    countries_wo_recovered_info = [country for country in all_countries if country not in countries_w_recovered_info]
    return countries_wo_recovered_info


def convert_columns_to_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    original = frame.columns.tolist()
    converted = pd.to_datetime(frame.columns[4:], format='%m/%d/%y')
    original[4:] = converted
    frame.columns = original
    return frame


def fill_recovered_by_infection_time(confirmed: pd.DataFrame,
                                     deaths: pd.DataFrame,
                                     recovered: pd.DataFrame,
                                     countries: List) -> pd.DataFrame:
    confirmed_temp = confirmed.loc[countries, :]
    deaths_temp = deaths.loc[countries, :]
    recovered_temp = confirmed_temp.copy()
    dates = confirmed_temp.columns.values[4:]
    for i, date in enumerate(dates):
        if i > 13:
            recovered_temp.loc[:, pd.to_datetime(date)] = confirmed_temp.loc[:, pd.to_datetime(date) - pd.DateOffset(
                days=14)] - deaths_temp.loc[:, pd.to_datetime(date) - pd.DateOffset(days=14)]
        else:
            recovered_temp.loc[:, pd.to_datetime(date)] = 0
    num = recovered_temp._get_numeric_data()
    num[num < 0] = 0
    recovered = recovered.append(recovered_temp)
    recovered = recovered.sort_index()
    recovered = recovered.drop(['Canada '], axis=0)
    return recovered


def get_mortality_info(frame: pd.DataFrame) -> List:
    last_date = frame.columns[-1]
    countries_wo_mortality_info = frame[frame[last_date] <= 0].index.values
    return countries_wo_mortality_info


def drop_countries_wo_mortality_info(confirmed: pd.DataFrame,
                                     deaths: pd.DataFrame,
                                     recovered: pd.DataFrame,
                                     countries: List) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    recovered = recovered.drop(countries, axis=0)
    deaths = deaths.drop(countries, axis=0)
    confirmed = confirmed.drop(countries, axis=0)
    return confirmed, deaths, recovered


def calculate_active_cases(confirmed: pd.DataFrame,
                           deaths: pd.DataFrame,
                           recovered: pd.DataFrame) -> pd.DataFrame:
    active_cases = confirmed.copy()
    dates = active_cases.columns.values[4:]
    for date in dates:
        active_cases.loc[:, pd.to_datetime(date)] = confirmed.loc[:, pd.to_datetime(date)] - \
                                                    deaths.loc[:, pd.to_datetime(date)] - \
                                                    recovered.loc[:, pd.to_datetime(date)]
    return active_cases


def strip_active_cases(confirmed: pd.DataFrame,
                       deaths: pd.DataFrame,
                       recovered: pd.DataFrame,
                       active: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    last_date = active.columns[-1]
    countries = active[active[last_date] <= 100].index.values
    active = active.drop(countries, axis=0)
    confirmed = confirmed.drop(countries, axis=0)
    deaths = deaths.drop(countries, axis=0)
    recovered = recovered.drop(countries, axis=0)
    return confirmed, deaths, recovered, active


def calculate_monthly_mortality(deaths: pd.DataFrame, recovered: pd.DataFrame):
    def transpose_and_add_date_column(frame: pd.DataFrame):
        new_frame = frame.T
        new_frame['Year/Month'] = new_frame.index.to_series().dt.strftime('%Y-%m')
        return new_frame

    daily_deaths = deaths.copy()
    daily_recoveries = recovered.copy()
    dates = daily_deaths.columns.values[4:]

    for i, date in enumerate(dates):
        if i > 1:
            previous_date = pd.to_datetime(date) - pd.DateOffset(days=1)
            daily_deaths.loc[:, pd.to_datetime(date)] = deaths.loc[:, pd.to_datetime(date)] - deaths.loc[:,
                                                                                              pd.to_datetime(
                                                                                                  previous_date)]

            daily_recoveries.loc[:, pd.to_datetime(date)] = recovered.loc[:, pd.to_datetime(date)] - recovered.loc[:,
                                                                                                     pd.to_datetime(
                                                                                                         previous_date)]
    daily_deaths = daily_deaths.drop(labels=daily_deaths.columns.values[:4], axis=1)
    daily_recoveries = daily_recoveries.drop(labels=daily_recoveries.columns.values[:4], axis=1)
    daily_deaths = transpose_and_add_date_column(daily_deaths)
    daily_recoveries = transpose_and_add_date_column(daily_recoveries)
    daily_deaths = daily_deaths.groupby(['Year/Month']).sum()
    daily_recoveries = daily_recoveries.groupby(['Year/Month']).sum()
    daily_deaths['Monthly mortality'] = daily_deaths.sum(axis=1)
    daily_recoveries['Monthly recoveries'] = daily_recoveries.sum(axis=1)
    daily_deaths['Accumulated monthly mortality'] = (daily_deaths['Monthly mortality'] / daily_recoveries['Monthly recoveries'])
    print(daily_deaths)



if __name__ == '__main__':
    # pandasgui.show(deaths_df, settings={'block': True})
    confirmed_df, deaths_df, recovered_df = read_data()
    confirmed_df = expand(confirmed_df)
    deaths_df = expand(deaths_df)
    recovered_df = expand(recovered_df)
    confirmed_df = convert_columns_to_datetime(confirmed_df)
    deaths_df = convert_columns_to_datetime(deaths_df)
    recovered_df = convert_columns_to_datetime(recovered_df)
    countries_wo_recovered_info = identify_countries_wo_recovered_info(confirmed_df, recovered_df)
    recovered_df = fill_recovered_by_infection_time(confirmed_df, deaths_df, recovered_df, countries_wo_recovered_info)
    countries_wo_mortality_info = get_mortality_info(deaths_df)
    confirmed_df, deaths_df, recovered_df = drop_countries_wo_mortality_info(confirmed_df, deaths_df, recovered_df,
                                                                             countries_wo_mortality_info)
    active_cases_df = calculate_active_cases(confirmed_df, deaths_df, recovered_df)
    confirmed_df, deaths_df, recovered_df, active_cases_df = strip_active_cases(confirmed_df, deaths_df, recovered_df,
                                                                                active_cases_df)

    calculate_monthly_mortality(deaths_df, recovered_df)

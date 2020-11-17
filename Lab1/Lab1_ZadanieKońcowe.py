import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandasgui

# Zadanie 1
df_population = pd.DataFrame(pd.read_csv("population_by_country_2019_2020.csv", index_col=0))
print(df_population)
print('-------------------')
# Zadanie 2
df_population_summary = df_population.describe()
print(df_population_summary)
print('-------------------')
pandasgui.show(df_population, settings={'block': True})
# Zadanie 3
df_population['Net population change'] = df_population['Population (2020)'] - df_population['Population (2019)']
df_population['Population change [%]'] = (df_population['Population (2020)'] - df_population['Population (2019)']) / \
                                         df_population['Population (2019)']
print(df_population)
print('-------------------')
pandasgui.show(df_population, settings={'block': True})
# Zadanie 4
df_population.sort_values(by='Population change [%]', ascending=False, inplace=True)
df_highest_change = df_population.iloc[0:10, :]
df_highest_change = df_highest_change.filter(regex='Population .20.*')
print(df_highest_change)
print('-------------------')
pandasgui.show(df_highest_change, settings={'block': True})
# Zadanie 5
df_highest_change.plot(kind='bar')
plt.show()
# Zadanie 6
df_population['Density (2020)'] = 'Low'
print(df_population)
print('-------------------')

# Zadanie 7
df_population.loc[df_population['Population (2020)'] / df_population['Land Area (Km²)'] > 500,
                  'Density (2020)'] = 'High'
pandasgui.show(df_population, settings={'block': True})

#Zadanie 8
df_to_save = df_population.iloc[0::2, :]
df_to_save.to_csv("saved.csv")
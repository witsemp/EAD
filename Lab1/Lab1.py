import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandasgui

num_days = 20
temperature = np.random.uniform(20, 28, size=(num_days, 1))
pressure = np.random.uniform(990, 1010, size=(num_days, 1))
rain = np.random.uniform(0, 20, size=(num_days, 1))
random_data = np.hstack((temperature, pressure, rain))
df_weather = pd.DataFrame(index=pd.date_range("20200501", periods=num_days, freq="1D"),
                          data=random_data, columns=["Temperature", "Pressure", "Rain"])
print(df_weather)
df_people = pd.DataFrame({"Height": [180, 160, 195],
                          "Weight": [77, 52, 200]})
# print(df_people)
df_weather_summary = df_weather.describe()
# print(df_weather_summary)
# df_weather.plot()
# plt.show()
# df_weather.plot(kind='scatter', x='Temperature', y='Rain')
# plt.show()
# pandasgui.show(df_weather, settings={'block': True})
vals = np.random.randn(6, 4)
df = pd.DataFrame(vals, index=[0.0, 0.2, 0.3, 0.7, 1.0, 1.3], columns=["A", "B", "C", "D"])
print(df)
# print(df.index)
# print(df.columns)
list_of_columns = list(df.columns)
# for c in list_of_columns:
#     print(c)
# print(df.values)
# print(df[0.2:1.0])
# print(df.loc[0.2:0.3, "A":"C"])
# print(df.loc[[0.0, 0.2, 0.3], :])
# print(df.loc[0:3, ["C"]])
# print(df.filter(regex=r"[A-C]"))
# print(df.loc[(df["A"] > -0.75) & (df["B"] < 0.25), :])
alpha = np.array([0, np.pi/4, np.pi/2, np.pi*3/4, np.pi])
trig = pd.DataFrame({"sinus": np.round(np.sin(alpha), 10),
                     "cosinus" : np.round(np.cos(alpha), 10),
                     "x^2" : alpha**2,
                     "random" : np.random.randn(len(alpha))}, index=alpha)
# print(trig)
# trig.loc[1:4, "random"] = 0
# print(trig.loc[trig["cosinus"] >= 0, "random"])
print(trig)
trig.set_index(trig["sinus"], inplace=True)
print(trig)
rename_dict = {"sinus": "simus"}
trig.rename(columns=rename_dict, inplace=True)
print(trig)
trig.sort_values("cosinus", axis=0, inplace=True, ascending=False)  # sortowanie wzdłuż osi 0 (po wierszach), w miejscu, malejąco
print(trig)
trig.to_csv("trig.csv")
new_trig = pd.DataFrame(pd.read_csv("trig.csv", index_col=0))
print(new_trig)

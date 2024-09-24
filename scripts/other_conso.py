import pandas as pd
import numpy as np
from datetime import datetime
import json

df2 = pd.read_csv('../data/TC3_15001_30-04-2014.csv')
df2['date'] = df2['Date and Time of capture'].apply(
    lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M:%S"))
df2['Date'] = df2['date'].apply(lambda x: x.date())
df2 = df2.sort_values(
    ['Location ID', 'date']).drop(columns='Date and Time of capture')

df_house = df2[df2['Measurement Description'] != 'heat pump power consumption']
df_pump = df2[df2['Measurement Description'] == 'heat pump power consumption']


def find_closest(i):
    index = 0
    min = np.inf
    for j in range(len(df_pump)):
        if (df_pump['date'].iloc[j] - df_house['date'].iloc[i]).seconds < min:
            min = (df_pump['date'].iloc[j] - df_house['date'].iloc[i]).seconds
            index = j
    return min, index


conso = np.zeros(len(df_house))
for i in range(len(df_house)):
    min, index = find_closest(i)
    conso[i] = df_house['Parameter'].iloc[i] - df_pump['Parameter'].iloc[index]

df_house['Conso'] = conso
conso_hour = []
for i in range(24):
    conso_hour.append(0)
    count = 0
    for j in range(len(df_house)):
        if df_house['date'].iloc[j].hour == i:
            count += 1
            conso_hour[-1] += df_house['Conso'].iloc[j]
    conso_hour[-1] /= count

with open("./data/TC3_clean", "w") as fp:
    json.dump(conso_hour, fp)

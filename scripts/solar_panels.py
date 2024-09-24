import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

path = "C:/Users/miche/Downloads/ninja_pv_51.7520_-1.2578_uncorrected.csv"
df = pd.read_csv(path, skiprows=3)

start_date = datetime(2019, 4, 30).date()
end_date = datetime(2019, 5, 2).date()

df['date'] = df['time'].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M").date())
df1 = df[df['date'] >= start_date]
df2 = df1[df1['date'] <= end_date]
weather_forecast = df2['temperature'].values
power_solar = df2['electricity']

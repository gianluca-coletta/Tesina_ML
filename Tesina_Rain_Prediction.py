import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


rain = pd.read_csv("weatherAUS.csv")

print(rain.head())
print(rain.info())
print(rain.describe())

NaN_label = rain['RainTomorrow'].isnull().sum()
print("Number of labels with a NaN value = " + str(NaN_label))
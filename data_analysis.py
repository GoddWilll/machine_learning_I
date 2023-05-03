#!/usr/bin/python3
#encoding:utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
							confusion_matrix, log_loss)


#importing data file and creating a DataFrame
df_train = pd.read_csv("data/train.csv")

X = df_train[df_train.columns.drop(["Y"])]

#data analysis
fig, ax = plt.subplots(3, 4, figsize=(11, 10), sharey=True)
for col, axis in zip(X.columns, ax.flatten()):
	sns.scatterplot(x=df_train[col], y=df_train["Y"], ax=axis)


fig, ax = plt.subplots(3, 4, figsize=(20, 10))
for col, axis in zip(X.columns, ax.flatten()):
	if col == "X11" or col == "X12":
		sns.boxplot(x=df_train[col], y=df_train["Y"], ax=axis)
	else:
		sns.boxplot(x=df_train["Y"], y=df_train[col], ax=axis)

plt.show()
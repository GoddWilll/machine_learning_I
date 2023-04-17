#!/usr/bin/python3
#encoding:utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.impute import SimpleImputer

#importing models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
							confusion_matrix, log_loss)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#importing data file and creating a DataFrame
df_train = pd.read_csv("data/train.csv")

#transforming categorical columns
df_train = df_train.astype({"X11":"category", "X12":"category"})

#updating null values foreach column with mean
df_train["X11"].fillna("T1", inplace=True)
df_train["X12"].fillna("T29", inplace=True)
for col in df_train.columns.drop(["X11", "X12"]):
	df_train[col].fillna(np.mean(df_train[col]), inplace=True)

#updating null values foreach column with mean
cont = SimpleImputer(missing_values=np.nan, strategy="mean")
cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

cont_cols = df_train.select_dtypes(exclude=["category"]).columns
cat_cols = df_train.select_dtypes(include=["category"]).columns

df_train[cont_cols] = cont.fit_transform(df_train[cont_cols])
df_train[cat_cols] = cat.fit_transform(df_train[cat_cols])

#retransforming categorical columns
df_train = df_train.astype({"X11":"category", "X12":"category"})

#consulting some infos
print(df_train.head())
print(df_train.info())
print("Sum of Null values in df:")
print(df_train.isna().sum())
print()

#selecting targets and splitting datasets
X = df_train[df_train.columns.drop(["Y"])]
Y = df_train[["Y"]]

X_train, X_test, Y_train, Y_test = train_test_split(
	X, Y, train_size=.8, test_size=.2, shuffle=True, random_state=0
)

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

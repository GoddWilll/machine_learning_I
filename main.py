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

#importing models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#importing data file and creating a DataFrame
df_train = pd.read_csv("data/train.csv")

numeric_features = ["X1", "X4", "X5", "X6", "X7"]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ["X12"]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

"""
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

#creating OHE for the 2 categorical cols
ohe = OneHotEncoder(drop='first')
df_train_cat = ohe.fit_transform(df_train[cat_cols])
df_train_cat = pd.DataFrame(df_train_cat.toarray())

df_train.reset_index(drop=True, inplace=True)
df_train = pd.concat([df_train[cont_cols], df_train_cat], axis=1)

#retransforming categorical columns
#df_train = df_train.astype({"X11":"category", "X12":"category"})

#consulting some infos
print(df_train.head())
print(df_train.info())
print("Sum of Null values in df:")
print(df_train.isna().sum())
print()
"""

#selecting targets and splitting datasets
X_sub = df_train[numeric_features+categorical_features]
cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
Y = cat.fit_transform(df_train[["Y"]])

# X_train, X_test, Y_train, Y_test = train_test_split(
# 	X, Y, train_size=.8, test_size=.2, shuffle=True, random_state=0
# )

"""
#data analysis
fig, ax = plt.subplots(3, 4, figsize=(11, 10), sharey=True)
for col, axis in zip(X_sub.columns, ax.flatten()):
	sns.scatterplot(x=df_train[col], y=df_train["Y"], ax=axis)


fig, ax = plt.subplots(3, 4, figsize=(20, 10))
for col, axis in zip(X_sub.columns, ax.flatten()):
	if col == "X11" or col == "X12":
		sns.boxplot(x=df_train[col], y=df_train["Y"], ax=axis)
	else:
		sns.boxplot(x=df_train["Y"], y=df_train[col], ax=axis)
"""

#creating model
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

model.fit(X_sub, Y)

pred_prob_train = pd.DataFrame(model.predict_proba(X_sub))
loss = log_loss(Y, pred_prob_train)
#print(f'Training log-loss : {loss}')

#importing test set
X_test = pd.read_csv("data/Xtest.csv")
pred_prob_test = pd.DataFrame(model.predict_proba(X_test))

pred_prob_test.rename(columns = {0:'Y_1', 1:'Y_2', 2:'Y_3', 3:'Y_4', 4:'Y_5', 5:'Y_6', 6:'Y_7'}, inplace=True)

#print(pred_prob_test.head())
#pred_prob_test.to_csv("../benchmark.csv", index=False)
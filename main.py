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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import threading
import sys

#importing data file and creating a DataFrame
df_train = pd.read_csv("data/train.csv")

numeric_features = ["X1", "X4", "X6", "X8", "X10"]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ["X11", "X12"]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


#selecting targets and splitting datasets
X_sub = df_train[numeric_features+categorical_features]
cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
Y = cat.fit_transform(df_train[["Y"]])

"""
class MyThread(threading.Thread):
	def __init__(self, ThreadName, X_sub, Y, preprocessor):
		super(MyThread, self).__init__()
		#creating model
		self.model = Pipeline(steps=[('preprocessor', preprocessor),
		                      ('classifier', ExtraTreesClassifier(n_estimators=1800, criterion="log_loss", max_features=None, n_jobs=4))])
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
			X_sub, Y, train_size=.8, test_size=.2, shuffle=True, random_state=0
		)
		self.threadName = ThreadName

	def run(self):
		self.model.fit(self.X_train, np.ravel(self.Y_train, order="C"))

		pred_prob_train = pd.DataFrame(self.model.predict_proba(self.X_test))
		loss = log_loss(self.Y_test, pred_prob_train)
		print(f'{self.threadName}: Training log-loss : {loss}')


thread_list = []

for i in range(3):
	x = MyThread(f"Thread-{i}", X_sub, Y, preprocessor)
	x.start()
	thread_list.append(x)

for t in thread_list:
	t.join()

sys.stdout.flush()


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

"""
#creating model
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', ExtraTreesClassifier(n_estimators=2400, criterion="log_loss", max_features=None, n_jobs=7))])

model.fit(X_sub, np.ravel(Y, order="C"))

pred_prob_train = pd.DataFrame(model.predict_proba(X_sub))
loss = log_loss(Y, pred_prob_train)
print(f'Training log-loss : {loss}')

#importing test set
X_test = pd.read_csv("data/Xtest.csv")
X_test_sub = X_test[numeric_features+categorical_features]
pred_prob_test = pd.DataFrame(model.predict_proba(X_test_sub))

print(pred_prob_test.head())

pred_prob_test.rename(columns = {0:'Y_1', 1:'Y_2', 2:'Y_3', 3:'Y_4', 4:'Y_5', 5:'Y_6', 6:'Y_7'}, inplace=True)
idx = pred_prob_test.index
pred_prob_test.insert(0, 'id', idx)

pred_prob_test.to_csv("../benchmark.csv", index=False)
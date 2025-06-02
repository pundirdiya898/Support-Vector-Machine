# IMPORT LIBRARY
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# GENERATE DATASET
from sklearn.datasets import make_regression
# without cofficient of underline model
X,y = make_regression(n_samples=500,n_features=5,coef=False,bias=12,noise=10,random_state=2529)
#with cofficient of underline model
X,y,w = make_regression(n_samples=500,n_features=5,coef=True,bias=12,noise=10,random_state=2529)
X.shape, y.shape
w #cofficient of X
# Get first five rows of feature (X) and target(y) variables.
X[0:5]
y[0:5]

# GET TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# Support Vector Machine Regression Model Train
from sklearn.svm import SVR
model = SVR()
model.fit(X_train,y_train)
SVR()

# GET MODEL PREDICTION
y_pred = model.predict(X_test)
y_pred.shape
y_pred

# GET MODEL EVALUATION
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
mean_squared_error(y_test,y_pred)
mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
r2_score(y_test,y_pred)

# Hyperparameter tunning : Grid Search
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':['linear','poly'], 'C' : [1,5,10]}
gridsearch = GridSearchCV(SVR(),parameters,scoring='neg_mean_absolute_error')
gridsearch.fit(X_train,y_train)
gridsearch.best_params_
gridsearch.best_score_
gridsearch.best_estimator_
gridsearch.best_index_
y_pred_grid = gridsearch.predict(X_test)
mean_absolute_error(y_test,y_pred_grid)
r2_score(y_test,y_pred_grid)

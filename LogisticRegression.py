# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:50:29 2022

@author: is-os
"""
#import libraries
import pandas as pd 
import numpy as np
#read csv file
dataset = pd.read_csv('Book12.csv',sep=';')
dataset.head(5)
#delete unnecessary columns  
dataset=dataset.drop(['default','contact','day','poutcome'],axis=1)
dataset.head(5)
# transform non-numerical labels to numerical labels
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
dataset.job=label.fit_transform(dataset.job)
dataset.marital=label.fit_transform(dataset.marital)
dataset.education=label.fit_transform(dataset.education)
dataset.housing=label.fit_transform(dataset.housing)
dataset.loan=label.fit_transform(dataset.loan)
dataset.month=label.fit_transform(dataset.month)
dataset.y=label.fit_transform(dataset.y)
dataset.head(5)

#replace the missing data if any 
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit(dataset)
dataset=imp.transform(dataset)
#split the data to work on training and testing later
X=dataset[:,:-1]
X.shape
Y=dataset[:,-1]
Y.shape
#scaling the data to be more faster and easy to work on
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X=std.fit_transform(X)
#now we split the data to training part and testing 70% training and 30% for testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=40)
X_train
X_test
Y_train
Y_test
X_train.shape
Y_train.shape
#predict the outcome 
from sklearn.linear_model import LogisticRegression
LRM = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=33)
LRM.fit(X_train, Y_train)
y_pred=LRM.predict(X_test)
#Scoring is 0.88 which is good
LRM.score(X_test, Y_test)
print('Accuracy Score is : ',LRM.score(X_test, Y_test))
y_pred_prob = LRM.predict_proba(X_test)

print('Predicted Value for LogisticRegressionModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for LogisticRegressionModel is : ' , y_pred_prob[:10])
#evaluate the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm


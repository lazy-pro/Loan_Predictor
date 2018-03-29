#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/shivam/loan_prediction/train.csv") 	#reading data into dataframe
y=df.Loan_Status
df = df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]

train_features,test_features,train_labels,test_labels = train_test_split(df, y, test_size=0.2)

#print train_features.shape
#print train_labels.shape
#print test_features.shape
#print test_labels.shape

from sklearn.ensemble import ExtraTreesClassifier
clf=ExtraTreesClassifier(n_estimators=30, min_samples_split=35,random_state=0)
clf.fit(train_features,train_labels)

pred = clf.predict(test_features)

from sklearn.metrics import accuracy_score
print accuracy_score(test_labels,pred)

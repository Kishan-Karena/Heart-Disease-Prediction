# Importing lib.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Processing
# loading the csv data 
heart=pd.read_csv("heart_disease.csv")

# print first 5 Rows of the dataset
heart.head()

# print last 5 Rows
heart.tail()

# Number of rows and columns in the dataset
heart.shape

# getting some info about the data
heart.info()

# checking for missing values
heart.isnull().sum()

# statistical measures about the data
heart.describe()

# checking the distribution of target variable
heart['target'].value_counts()

# 1--Defective Heart
# 0-Healthy Heart

# Splitting the Features and Target
x=heart.drop(columns='target',axis=1)
y=heart['target']
print(x)
print(y)

# Splitting the data into Tarining data & Test data
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

print(x.shape,X_train.shape,X_test.shape)

# Model Training 

model=LogisticRegression()
# training the LogisticRegression model with Traing data
model.fit(X_train,Y_train)

# Model Evaluation
# Accuracy Score

X_train_pred=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_pred,Y_train)

print("Accuracy on train data :",training_data_accuracy)

# Accuracy on test data
X_test_pred=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_pred,Y_test)

print("Accuracy on test data :",test_data_accuracy) 



# Building Predictive System
input_data=(62,0,0,140,268,0,0,160,0,3.6,0,2,2)
# change the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)
# reshape the numpy array 
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The Person dose not have a Heart Disease")
else:
    print("The Person has Heart Disease")
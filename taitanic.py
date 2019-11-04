import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data Preprocessing
train=pd.read_csv("train.csv")
x_train=train.iloc[:,2:12]
x_train=x_train.drop(["Name","Ticket","Cabin","Fare"],axis=1)
y_train=train.iloc[:,1:2]
test=pd.read_csv("test.csv")
x_test=test.iloc[:,1:11]
x_test=x_test.drop(["Name","Ticket","Cabin","Fare"],axis=1)

#missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(x_train.iloc[:, 2:3])
x_train.iloc[:, 2:3] = imputer.transform(x_train.iloc[:, 2:3])

most_frequent_embarked = x_train['Embarked'].value_counts().index[0]
x_train['Embarked'].fillna(most_frequent_embarked, inplace = True)

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(x_test.iloc[:, 2:3])
x_test.iloc[:, 2:3] = imputer.transform(x_test.iloc[:, 2:3])

x_train.info()

#categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_train = LabelEncoder()
x_train.iloc[:, 1] = labelencoder_x_train.fit_transform(x_train.iloc[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
x_train = onehotencoder.fit_transform(x_train).toarray()

labelencoder_x_test = LabelEncoder()
x_test.iloc[:, 1] = labelencoder_x_train.fit_transform(x_test.iloc[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
x_test = onehotencoder.fit_transform(x_test).toarray()

x_train.iloc[:,5] = labelencoder_x_train.fit_transform(x_train.iloc[:,5])
onehotencoder = OneHotEncoder(categorical_features = [5])
x_train = onehotencoder.fit_transform(x_train).toarray()

x_test.iloc[:,5] = labelencoder_x_test.fit_transform(x_test.iloc[:,5])
onehotencoder = OneHotEncoder(categorical_features = [5])
x_test = onehotencoder.fit_transform(x_test).toarray()

x_train=x_train[:,1:]
x_test=x_test[:,1:]
"""
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
"""

from sklearn.svm import SVC
classifier=SVC(kernel="linear")
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

subsmt=pd.DataFrame({"PassengerID":test["PassengerId"],"survived":y_pred})
subsmt.to_csv("subsmt.csv",index=False)



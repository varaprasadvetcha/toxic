import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Churn_Modelling.csv")
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


import keras

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units=6,activation='relu',bias_initializer="uniform",input_dim=11))
classifier.add(Dense(units=6,activation='relu',bias_initializer="uniform"))
classifier.add(Dense(units=1,activation='sigmoid',bias_initializer="uniform"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(x_train,y_train,batch_size=10,epochs=100)

y_pred=classifier.predict(x_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)







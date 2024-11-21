# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:50:46 2024

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dataset = pd.read_csv(r"D:\NareshIT\NIT DataSci Notes\NIT DataSci Notes\44. 23rd XGBOOST\7.XGBOOST\Churn_Modelling.csv")

X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],  # Apply OneHotEncoder to column 1 
    remainder='passthrough' # Leave other columns unchanged
    ) 
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


ann = tf.keras.models.Sequential()
ann

ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #input layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #hidden layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #output layer

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=25)

ann.save(r"D:\NareshIT\Spyder_Homework\AI\ANN\ann_churn.h5")

import pickle
filename = 'scalar.pkl'
with open(filename, 'wb') as file: pickle.dump(sc, file)

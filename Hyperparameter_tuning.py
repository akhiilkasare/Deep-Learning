# -*- coding: utf-8 -*-


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/akhil/Downloads/machine_learning/hyper_parameter_optimization/Churn_Modelling.csv')
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(x["Geography"],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

## Concatenate the Data Frames

x=pd.concat([x,geography,gender],axis=1)

## Drop Unnecessary columns
x=x.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Performing hyperparameter tuning

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  GridSearchCV
from keras.layers import Dropout

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LeakyReLU, Flatten, BatchNormalization
from keras.activations import relu, sigmoid

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        #This means that for the first layer you always have to give imput dimension to the particular neuron i.e. the Dense neuron
        if i == 0:
            # First hidden layer
            
            model.add(Dense(nodes, input_dim=x_train.shape[1])) 
           
            # What kind of activation function you need to give to the hidden layers
            
            model.add(Activation(activation))
            
            # If you want to apply any dropout rate
            
            model.add(Dropout(0.3))
            
        else:
            model.add(Dense())
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
            
        # Initializing the last layer
        
        model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)

'''
For each iteration

[[it will have 1 hideen layer with 20 hidden neurons], [two hidden layers 1st. layer will have 40 hidden layer and 2nd layer will have 20 hidden neuron],
[three hidden layers 1st.hidden layer will have 45 hidden neuron 2nd.hidden layer will have 30 hidden neuron 3rd. will have 15 hidden neuron]]
'''
layers = [[20], [40,20], [45,30,15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size=[128, 256], epochs=[30])

#cv=5 means i have to perform coss validation of 5 experiments
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

grid_result = grid.fit(x_train, y_train)

#Models best result
print(grid_result.best_score_, grid_result.best_params_)

pred_y = grid.predict(x_test)
y_pred = (pred_y > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)











# -*- coding: utf-8 -*-
"""Obesity Dataset


This dataset contains information about the obesity classification of individuals. The data was collected from a variety of sources, including medical records, surveys, and self-reported data. The dataset includes the following columns:

ID: A unique identifier for each individual \\
Age: The age of the individual \\
Gender: The gender of the individual \\
Height: The height of the individual in centimeters \\
Weight: The weight of the individual in kilograms \\
BMI: The body mass index of the individual, calculated as weight divided by height squared \\
Label: The obesity classification of the individual, which can be one of the following:
* Normal Weight
* Overweight
* Obese
* Underweight

110 samples that can be found in https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset.
"""

!gdown 1Rut46dX0xY9yqWRk1sqFGG8vRdTMdd8R

import pandas as pd


# Load the CSV data into a DataFrame usin Pandas
ob_data = pd.read_csv('/content/Obesity Classification.csv', index_col='ID')

# Display the first few rows of the DataFrame
print(ob_data.head())

# Check any missing values
print("There are {} missing values (NA) in the DataFrame".
      format(ob_data.isnull().sum().sum()))

# Feature encoding of the "Gender" attribute
ob_data.Gender = ob_data.Gender.map(lambda x: 1 if x=='Male' else -1)
print(ob_data.head())

# Check the suggested charts
ob_data

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from SFW import SFW_NN
from AdaSFW import AdaSFW_NN
from AdaSVRF import AdaSVRF_NN

X = ob_data.iloc[:,:-1].values
y = ob_data.iloc[:,-1].values

num_classes = len(set(y)) # Number of classification labels

# Binarize the outputs
y = LabelBinarizer().fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""### 2.2.1 SFW:"""

SFW_obesity = SFW_NN(X_train, y_train, X_test, y_test, initfactor = 20)
SFW_obesity.train()
SFW_obesity.test()
SFW_obesity.plot_learning_curves()

SFW_obesity.plot_2D_update()

"""### 2.2.2 AdaSFW

"""

AdaSFW_obesity = AdaSFW_NN(X_train, y_train, X_test, y_test, initfactor = 20)
AdaSFW_obesity.train()
AdaSFW_obesity.test()
AdaSFW_obesity.plot_learning_curves()

AdaSFW_obesity.plot_2D_update()

"""### 2.2.3 AdaSVRF"""

AdaSVRF_obesity = AdaSVRF_NN(X_train, y_train, X_test, y_test, initfactor = 20, nu = 10**(-1/2), K = 5)
AdaSVRF_obesity.train()
AdaSVRF_obesity.test()
AdaSVRF_obesity.plot_learning_curves()

AdaSVRF_obesity.plot_2D_update()

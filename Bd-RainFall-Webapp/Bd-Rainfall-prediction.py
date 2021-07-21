# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('Bd-Rainfall-prediction.csv')

# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='Rainfall(0=No,1=Yes)')
y = df['Rainfall(0=No,1=Yes)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'rainfall-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
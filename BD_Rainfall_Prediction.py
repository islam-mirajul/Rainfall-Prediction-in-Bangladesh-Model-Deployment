import numpy as np
import pandas as pd

dataset = pd.read_csv('Bd-Rainfall-prediction.csv')

dataset.describe()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

# Model training
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


#made prediction using new data

prediction_1 = classifier.predict(sc.transform(np.array([[13,2.5,2,75,60,5.7,4.8,300]])))
prediction_1_proba = classifier.predict_proba(sc.transform(np.array([[13,2.5,2,75,60,5.7,4.8,300]])))[:,1]

prediction_2 = classifier.predict(sc.transform(np.array([[27,26,1,90,94,4.6,5.6,275]])))
prediction_2_proba = classifier.predict_proba(sc.transform(np.array([[27,26,1,90,94,4.6,5.6,275]])))[:,1]




# Picking the Model and Standard Scaler

import pickle
model_file = "RandomForest.pickle"
pickle.dump(classifier, open(model_file,'wb'))
scaler_file = "scaller.pickle"
pickle.dump(sc, open(scaler_file,'wb'))

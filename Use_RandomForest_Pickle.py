import pickle
import numpy as np

Use_classifier = pickle.load(open('RandomForest.pickle','rb'))
Use_scaler = pickle.load(open('scaller.pickle','rb'))


#made prediction using new data

prediction_1 = Use_classifier.predict(Use_scaler.transform(np.array([[13,2.5,2,75,60,5.7,4.8,300]])))
prediction_1_proba = Use_classifier.predict_proba(Use_scaler.transform(np.array([[13,2.5,2,75,60,5.7,4.8,300]])))[:,1]


prediction_2 = Use_classifier.predict(Use_scaler.transform(np.array([[27,26,1,90,94,4.6,5.6,275]])))
prediction_2_proba = Use_classifier.predict_proba(Use_scaler.transform(np.array([[27,26,1,90,94,4.6,5.6,275]])))[:,1]

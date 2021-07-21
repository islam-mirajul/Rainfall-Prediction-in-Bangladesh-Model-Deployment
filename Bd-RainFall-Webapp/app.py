from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier 
filename = 'rainfall-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        maxtemp = int(request.form['max_temp'])
        mintemp = int(request.form['min_temp'])
        aev = int(request.form['actual_evaporation'])
        rh9 = int(request.form['relative_humidity(9.00am)'])
        rh2 = int(request.form['relative_humidity(2.00pm)'])
        sun = float(request.form['sunshine'])
        cld = float(request.form['cloudy'])
        sr = int(request.form['solar_radiation'])
        
        data = np.array([[maxtemp, mintemp, aev, rh9, rh2, sun, cld, sr]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
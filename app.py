import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Create the Flask app
app = Flask(__name__)

# Load the pkl file
model = pickle.load(open('model.pkl','rb'))

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Function
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if not os.path.exists('img_data'):
        os.makedirs('img_data')
         
    output = round(prediction[0],2)

    return render_template('index.html', prediction_text='Employee Salary Should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

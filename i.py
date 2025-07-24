from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from ultralytics import YOLO



# Create a Flask app
app = Flask(__name__)

# Load the trained model
# with open('model\\20kbest.pt', 'rb') as f:
#     model = pickle.load(f)
model = YOLO('model/20kbest.pt')
# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data
    data = request.get_json()
    features = np.array(data['features'])

    # Make predictions
    prediction = model.predict(features.reshape(1, -1))

    # Return the prediction
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
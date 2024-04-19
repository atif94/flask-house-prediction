from flask import Flask, render_template, request
import pickle
import numpy as np
from credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

app = Flask(__name__)
app.config['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
app.config['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'model.pkl' file not found. Please run 'model.py' first to create the model.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input feature values from the form
    feature_values = [float(x) for x in request.form.values()]

    # Convert feature values to a NumPy array
    feature_array = np.array(feature_values)

    # Reshape the feature array to match the model's expected input shape
    feature_final = feature_array.reshape(1, -1)

    # Print the shape and data type of the input features
    print("Input shape:", feature_final.shape)
    print("Input data type:", feature_final.dtype)

    try:
        # Make the prediction
        prediction = model.predict(feature_final)
        prediction_text = f"Price of House will be Rs. {int(prediction[0][0])}"
    except Exception as e:
        prediction_text = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
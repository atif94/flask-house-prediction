from flask import Flask, render_template, request
import pickle
import numpy as np
from credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

app = Flask('__name__')
app.config['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
app.config['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    feature = [int(x) for x in request.form.values()]
    feature_final = np.array(feature).reshape(-1, 1)
    prediction = model.predict(feature_final)
    return render_template('index.html', prediction_text='Price of House will be Rs. {}'.format(int(prediction)))

if __name__ == '__main__':
    app.run(debug=True)
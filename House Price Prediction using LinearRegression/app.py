import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)
# Load the model
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    # Here I just want to transform and reshape my data
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    # here I will predict my data
    output = model.predict(new_data)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    output = model.predict(final_input)[0]
    return render_template("index.html", prediction_text="The House Price Prediction is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)
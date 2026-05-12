from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get input values
        features = [float(x) for x in request.form.values()]

        # Convert to numpy array
        final_input = np.array(features).reshape(1, -1)

        # Scale input
        final_input = scaler.transform(final_input)

        # Predict
        prediction = model.predict(final_input)

        # Result
        result = "Malignant Cancer" if prediction[0] == 0 else "Benign Cancer"

        return render_template(
            'index.html',
            prediction_text=f'Prediction Result: {result}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
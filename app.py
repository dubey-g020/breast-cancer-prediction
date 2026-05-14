from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ============================================================
# HOME PAGE
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # ====================================================
        # GET INPUT VALUES
        # ====================================================

        features = [float(x) for x in request.form.values()]

        # ====================================================
        # CONVERT TO NUMPY ARRAY
        # ====================================================

        final_input = np.array(features).reshape(1, -1)

        # ====================================================
        # SCALE INPUT
        # ====================================================

        final_input = scaler.transform(final_input)

        # ====================================================
        # PREDICT
        # ====================================================

        prediction = model.predict(final_input)

        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        probability = model.predict_proba(final_input)[0]

        # ====================================================
        # CONFIDENCE SCORE
        # ====================================================

        confidence = max(probability) * 100

        # ====================================================
        # RESULT
        # ====================================================

        result = (
            "Malignant Cancer"
            if prediction[0] == 0
            else "Benign Cancer"
        )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return render_template(
            'index.html',
            prediction_text=f'Prediction Result: {result}',
            confidence_text=f'Confidence: {confidence:.2f}%'
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)
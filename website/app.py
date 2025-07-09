from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and preprocessing tools
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
model = joblib.load(os.path.join(MODEL_DIR, 'final_obesity_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.pkl'))
feature_encoders = joblib.load(os.path.join(MODEL_DIR, 'feature_encoders.pkl'))
target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": " Obesity Risk Prediction API is live.",
        "endpoints": {
            "/predict": "POST request with JSON body to get obesity risk class"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # 🔧 Convert numeric values from strings to float
        numeric_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        for col in numeric_columns:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(float)

        #  Convert height from cm to meters if needed
        if df_input["Height"].max() > 10:
            df_input["Height"] = df_input["Height"] / 100.0

        #  BMI Calculation (before encoding)
        weight = df_input["Weight"].values[0]
        height = df_input["Height"].values[0]
        bmi = weight / (height ** 2)

        # Encode categorical features
        for col, encoder in feature_encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col] = encoder.transform(df_input[col])
                except Exception as e:
                    return jsonify({
                        "error": f"Invalid value for column '{col}': {df_input[col].values[0]}",
                        "expected": list(encoder.classes_)
                    }), 400
            else:
                return jsonify({"error": f"Missing required field: {col}"}), 400

        # Add missing columns if any
        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[feature_columns]

        #  Scale
        input_scaled = scaler.transform(df_input)

        #  Predict
        pred_encoded = model.predict(input_scaled)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

        #  Override prediction if BMI is normal
        if pred_label == "Insufficient_Weight" and 18.5 <= bmi <= 24.9:
            pred_label = "Normal_Weight"

        return jsonify({
            "prediction": pred_label,
            "bmi": round(bmi, 2),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
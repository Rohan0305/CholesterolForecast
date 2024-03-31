# Your Flask backend code with improvements

# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats

# Create Flask app
app = Flask(__name__)
CORS(app)

# Define endpoint for cholesterol prediction
@app.route('/predict', methods=['POST'])
def predict_cholesterol():
    try:
        # Load data from CSV
        data = pd.read_csv('backend/heart_health.csv')

        # Preprocessing
        data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure(mmHg)'].str.split('/', expand=True)
        data['Systolic_BP'] = data['Systolic_BP'].astype(float)
        data['Diastolic_BP'] = data['Diastolic_BP'].astype(float)
        data['Smoker'] = data['Smoker'].map({'Yes': 1, 'No': 0})
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
        data_numeric = data.drop(columns=['ID', 'Name', 'Heart Attack'])

        # Define features and target
        features = ['Age', 'Height(cm)', 'Weight(kg)', 'Systolic_BP', 'Diastolic_BP', 'Glucose(mg/dL)', 'Exercise(hours/week)', 'Smoker', 'Gender']
        X = data_numeric[features]
        y = data_numeric['Cholesterol(mg/dL)']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Data scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Prediction
        new_data = pd.DataFrame([request.json], columns=features)
        new_data_scaled = scaler.transform(new_data)
        predicted_cholesterol = model.predict(new_data_scaled)[0]

        # Confidence interval calculation
        confidence_level = 0.95
        y_pred = model.predict(X_test_scaled)
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
        z_score = scipy.stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin_of_error = z_score * residual_std
        lower_bound = predicted_cholesterol - margin_of_error
        upper_bound = predicted_cholesterol + margin_of_error

        # Prepare response
        response = {
            'predicted_cholesterol': predicted_cholesterol,
            'confidence_interval': [lower_bound, upper_bound]
        }
        return jsonify(response)

    except FileNotFoundError:
        return jsonify({'error': 'File not found. Please check the file path and try again.'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)




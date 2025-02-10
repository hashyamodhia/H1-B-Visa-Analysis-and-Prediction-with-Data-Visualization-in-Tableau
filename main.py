import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the trained model and encoders
with open("H1B Visa Model.pkl", "rb") as file:
    visa_model = pickle.load(file)

with open("case_status.pkl", "rb") as file:
    case_status_encoder = pickle.load(file)

with open("agent_representing_employer_encoder.pkl", "rb") as file:
    agent_encoder = pickle.load(file)

with open("full_time_position_encoder.pkl", "rb") as file:
    full_time_position_encoder = pickle.load(file)

with open("H1B_dependent_encoder.pkl", "rb") as file:
    dependent_encoder = pickle.load(file)

with open("occupation_encoder.pkl", "rb") as file:
    occupation_encoder = pickle.load(file)

with open("wage_scaled.pkl", "rb") as file:
    wage_scaler = pickle.load(file)

with open("willfull_violator_encoder.pkl", "rb") as file:
    violator_encoder = pickle.load(file)

with open("worksite_state_encoder.pkl", "rb") as file:
    worksite_state_encoder = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_value', methods=["POST"])
def predict_value():
    # Retrieve and process form data
    FULL_TIME_POSITION = request.form.get("FULL_TIME_POSITION")
    TOTAL_WORKER_POSITIONS = request.form.get("TOTAL_WORKER_POSITIONS")
    AGENT_REPRESENTING_EMPLOYER = request.form.get("AGENT_REPRESENTING_EMPLOYER")
    WORKSITE_STATE = request.form.get("WORKSITE_STATE")
    H1B_DEPENDENT = request.form.get("H1B_DEPENDENT")
    WILLFUL_VIOLATOR = request.form.get("WILLFUL_VIOLATOR")
    OCCUPATION = request.form.get("OCCUPATION")
    EMPLOYMENT_DURATION_YEARS = float(request.form.get("EMPLOYMENT_DURATION_YEARS"))
    PREVAILING_WAGE = float(request.form.get("PREVAILING_WAGE_SCALED"))

    # Safe transformation function for encoding
    def safe_transform(encoder, value):
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return -1  # Default for unseen labels

    # Encode categorical variables safely
    FULL_TIME_POSITION_ENCODED = safe_transform(full_time_position_encoder, FULL_TIME_POSITION)
    AGENT_REPRESENTING_EMPLOYER_ENCODED = safe_transform(agent_encoder, AGENT_REPRESENTING_EMPLOYER)
    WORKSITE_STATE_ENCODED = safe_transform(worksite_state_encoder, WORKSITE_STATE)
    H1B_DEPENDENT_ENCODED = safe_transform(dependent_encoder, H1B_DEPENDENT)
    WILLFUL_VIOLATOR_ENCODED = safe_transform(violator_encoder, WILLFUL_VIOLATOR)
    OCCUPATION_ENCODED = safe_transform(occupation_encoder, OCCUPATION)

    # Scale the prevailing wage correctly
    PREVAILING_WAGE_SCALED = wage_scaler.transform(np.array([[PREVAILING_WAGE]]))[0][0]  # Scale as a 2D array

    # Prepare input for prediction
    input_features = [
        FULL_TIME_POSITION_ENCODED,
        TOTAL_WORKER_POSITIONS,
        AGENT_REPRESENTING_EMPLOYER_ENCODED,
        WORKSITE_STATE_ENCODED,
        H1B_DEPENDENT_ENCODED,
        WILLFUL_VIOLATOR_ENCODED,
        OCCUPATION_ENCODED,
        EMPLOYMENT_DURATION_YEARS,
        PREVAILING_WAGE_SCALED
    ]

    # Make prediction
    prediction = visa_model.predict([input_features])[0]

    # Interpret result

    if prediction[0] == 0:
        result = "Visa Certified"
    elif prediction[0] == 1:
        result = "Visa Certified-Withdrawn"
    elif prediction[0] == 3:
        result = "Visa Withdrawn"
    else:
        result = "Denied"
    return f"<h1>Prediction: {result}</h1>"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

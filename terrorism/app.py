from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open("terrorism_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
# Function to preprocess input
def preprocess_input(killed, wounded, target_type, weapon_type, country, year, group_size):
    input_data = pd.DataFrame({
        'Killed': [killed],
        'Wounded': [wounded],
        'Target_type': [target_type],
        'Weapon_type': [weapon_type],
        'Country': [country],
        'Year': [year],
        'Group': [group_size]
    })

    # Apply one-hot encoding
    input_data = pd.get_dummies(input_data, columns=['Target_type', 'Weapon_type', 'Country'])

    # Ensure all features are present
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0  # Assign 0 for missing features

    # Ensure correct column order
    input_data = input_data[feature_names]

    # Scale the input
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        killed = int(request.form["killed"])
        wounded = int(request.form["wounded"])
        target_type = request.form["target_type"]
        weapon_type = request.form["weapon_type"]
        country = request.form["country"]
        year = int(request.form["year"])
        group_size = int(request.form["group_size"])

        # Preprocess input data
        processed_data = preprocess_input(killed, wounded, target_type, weapon_type, country, year, group_size)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Convert numerical prediction to attack type
        attack_types = {1: "Bombing/Explosion", 2: "Assassination", 3: "Unarmed Assault", 4: "Armed Assault",
                        5: "Facility/Infrastructure Attack", 6: "Hostage Taking", 7: "Hijacking", 8: "Unknown", 9: "Barricade Incident"}
        predicted_attack = attack_types.get(prediction, "Unknown Attack Type")

        return render_template("index.html", prediction_text=f"Predicted Attack Type: {predicted_attack}")

if __name__ == "__main__":
    app.run(debug=True)

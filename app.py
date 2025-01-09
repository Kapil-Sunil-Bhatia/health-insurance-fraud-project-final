from flask import Flask, request, render_template, jsonify
import pandas as pd
import io

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route to display the HTML form
@app.route("/")
def home():
    return render_template("index.html")  # This renders the HTML form

# Route to handle the file upload and prediction
@app.route("/predictfraud", methods=["POST"])
def predict_datapoint():
    # Check if all files and provider_id are in the request
    if "Provider" not in request.files or "Inpatient" not in request.files or "Outpatient" not in request.files or "Beneficiary" not in request.files:
        return jsonify({"error": "Please upload Inpatient, Outpatient, and Beneficiary CSV files."}), 400

    provider_id = request.form.get("provider_id")
    if not provider_id:
        return jsonify({"error": "Please provide a 'provider_id' string in the form data."}), 400

    # Read the CSV files from the request
    try:
        provider_file = request.files["Provider"]
        inpatient_file = request.files["Inpatient"]
        outpatient_file = request.files["Outpatient"]
        beneficiary_file = request.files["Beneficiary"]

        # Convert the files into pandas DataFrames
        provider_df = pd.read_csv(io.StringIO(provider_file.read().decode("utf-8")))
        inpatient_df = pd.read_csv(io.StringIO(inpatient_file.read().decode("utf-8")))
        outpatient_df = pd.read_csv(io.StringIO(outpatient_file.read().decode("utf-8")))
        beneficiary_df = pd.read_csv(io.StringIO(beneficiary_file.read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": f"Error reading CSV files: {str(e)}"}), 500

    try:
        # Preprocess data using CustomData class
        data = CustomData(provider_id, provider_df, inpatient_df, outpatient_df, beneficiary_df)
        data_preprocessed = data.get_preprocessed_data()

        # Predict using the trained model
        predict_pipeline = PredictPipeline()
        predicted_class_label = predict_pipeline.predict(data_preprocessed)
        return jsonify({"predictions": predicted_class_label})
    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

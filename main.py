"""
File: main.py
Author: Eser Inan Arslan
Email: eserinanarslan@gmail.com
Description: Description: This file contains the code for running and forecasting with the model developed for Kenvue.
"""

import pandas as pd
from flask import jsonify

import src.util
import flask
import json
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.4f}'.format)

# Read configuration from the config file
config_path = "config.ini"
config = src.util.read_config(config_path)
# Get values from the config file
results_path = config.get("Settings", "results_path")
data = src.util.read_data(results_path)
# 'Date' sütunu oluşturma ve değerleri atama

def column_format(data):
    try:
        data['XGB_Predictions'] = data['XGB_Predictions'].round().astype(int)
        data['Arima_Predictions'] = data['Arima_Predictions'].round().astype(int)
        data['LSTM_Predictions'] = data['LSTM_Predictions'].round().astype(int)
        data = data.rename(columns={'Count': 'Actual_Count'})

    except Exception as e:
        print(f"Error formatting columns: {e}")
        # Handle the error as needed, e.g., exit the program or set default values

    return data

# Apply column formatting
data = column_format(data)

# Convert DataFrame to JSON
try:
    df = data.to_json(orient="records")
    df = json.loads(df)

except Exception as e:
    print(f"Error converting DataFrame to JSON: {e}")
    # Handle the error as needed, e.g., exit the program or set default values

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Define API endpoint
@app.route('/all_results', methods=['GET'])
def total_api():
    return jsonify(df[:100])

# Run the Flask app
try:
    app.run(host=config["Service"]["Host"], port=int(config["Service"]["Port"]), debug=True)
except Exception as e:
    print(f"Error running Flask app: {e}")
    # Handle the error as needed, e.g., exit the program or set default values

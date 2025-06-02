import pandas as pd
import numpy as np
import RPi.GPIO as GPIO
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

RELAY_PIN = 26  # BCM pin 26 (GPIO26), adjust as per your wiring
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

model = tf.keras.models.load_model("gru_model.h5")
scaler = joblib.load("scaler.pkl")  # scaler saved using joblib.dump(scaler, 'scaler.pkl')

def preprocess_data():
    water_usage = pd.read_csv('Water Usage/SOC Outlet-1/Table.csv',
                               usecols=["Date", "Time", "Device Name", "water_flow_in_cubic_meter"])
    data = water_usage[::-1]  # Reverse

    data["Timestamp"] = pd.to_datetime(data["Date"] + " " + data["Time"], format="%d-%m-%Y %H:%M:%S")
    data["Rounded_Timestamp"] = data["Timestamp"].dt.round("6H")
    data.set_index("Rounded_Timestamp", inplace=True)

    data.drop(columns=["Date", "Time", "Timestamp", "Device Name"], inplace=True)
    data = data.resample("6H").mean()

    data.rename(columns={'water_flow_in_cubic_meter': 'Water_Consumption'}, inplace=True)
    data.interpolate(method="linear", inplace=True)
    return data

def prepare_input_from_data(df):
    seq_length = 36				#last 36 entries are used as input sequence
    last_values = df['Water_Consumption'].values[-seq_length:]
    last_values = last_values.reshape(-1, 1)
    scaled_sequence = scaler.transform(last_values)
    return np.expand_dims(scaled_sequence, axis=0)

try:
    df = preprocess_data()
    input_sequence = prepare_input_from_data(df)
    predicted_flow = model.predict(input_sequence)[0][0]
    print(f"Predicted flow: {predicted_flow:.2f}")

    # Custom current flow value to simulate
    current_flow = float(input("Enter simulated flow value (cubic meter): "))

    if current_flow > predicted_flow:
        print("Flow is higher than expected. Turning pump OFF.")
        GPIO.output(RELAY_PIN, GPIO.LOW)  # OFF (depends on relay logic)
    else:
        print("Flow is acceptable. Turning pump ON.")
        GPIO.output(RELAY_PIN, GPIO.HIGH)  # ON

    time.sleep(10)  # Wait 10s to observe the result

except Exception as e:
    print("Error occurred:", e)
finally:
    GPIO.cleanup()

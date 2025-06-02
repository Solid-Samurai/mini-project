import socket
import time
import ssl
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ========== Adafruit IO Setup ==========
AIO_SERVER = "io.adafruit.com"
AIO_PORT = 8883  # MQTT over SSL
AIO_USERNAME = "Harish06"
AIO_KEY = "aio_RnSN07QPJWYkUo3scjPon4jQXgpP"

AIO_FEED = AIO_USERNAME + "/feeds/waterflow"

# ========== ESP32 TCP Setup ==========
#ESP32_IP = "192.168.216.91"  
#ESP32_PORT = 5000

# ========== GRU Model Setup ==========

# Load and process the dataset
dataset = pd.read_csv('Table.csv', usecols=["Date", "Time", "Device Name", "water_flow_in_cubic_meter"])
dataset = dataset[::-1]
dataset["Timestamp"] = pd.to_datetime(dataset["Date"] + " " + dataset["Time"], format="%d-%m-%Y %H:%M:%S")
dataset["Rounded_Timestamp"] = dataset["Timestamp"].dt.round("6H")
dataset.set_index("Rounded_Timestamp", inplace=True)
dataset.drop(columns=["Date", "Time", "Timestamp", "Device Name"], inplace=True)
dataset = dataset.resample("6H").mean()
dataset.rename(columns={'water_flow_in_cubic_meter': 'Water_Consumption'}, inplace=True)
dataset.interpolate(method="linear", inplace=True)

# Fit scaler
scaler = MinMaxScaler()
scaler.fit(dataset[['Water_Consumption']])

# Load GRU model
model = load_model("gru_model.keras")

# Prepare the last 10 entries from dataset
last_10_entries = dataset[-10:]['Water_Consumption'].values.reshape(-1, 1)
scaled_last_10 = scaler.transform(last_10_entries)
input_seq = np.reshape(scaled_last_10, (1, 10, 1))  # (batch_size, time_steps, features)

# Predict future flow
predicted_scaled_flow = model.predict(input_seq, verbose=0)
predicted_flow = scaler.inverse_transform([[predicted_scaled_flow[0][0]]])[0][0]

print(f"Predicted Future Flow (based on dataset): {predicted_flow}")

# ========== Global Variables ==========
mqtt_connected = False
sock = None
latest_sensor_flow = None

# ========== MQTT Callbacks ==========

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        print("Connected to Adafruit IO MQTT broker!")
        client.subscribe(AIO_FEED)
        mqtt_connected = True
    else:
        print(f"Failed to connect to MQTT broker, return code {rc}")

def on_message(client, userdata, msg):
    global latest_sensor_flow
    payload = msg.payload.decode()
    try:
        latest_sensor_flow = float(payload)
        print(f"Received Live Sensor Flow: {latest_sensor_flow}")
    except ValueError:
        print("Invalid sensor flow received!")

# ========== TCP Communication with ESP32 ==========

def connect_to_esp32():
    global sock
    HOST = "0.0.0.0"
    PORT = 5000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Waiting for connection from ESP32...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    sock=conn

def send_command_to_esp32(command):
    global sock
    if sock:
        try:
            sock.sendall((command + "\n").encode())
            print(f"Sent command to ESP32: {command}")
        except socket.error:
            print("Failed to send command to ESP32, reconnecting...")
            sock.close()
            connect_to_esp32()

# ========== Main ==========
def main():
    global sock, latest_sensor_flow

    # Initialize MQTT Client
    client = mqtt.Client()
    client.username_pw_set(AIO_USERNAME, AIO_KEY)
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to Adafruit IO
    print("Connecting to Adafruit IO MQTT broker...")
    client.connect(AIO_SERVER, AIO_PORT, keepalive=60)

    # Start MQTT loop in background
    client.loop_start()

    # Connect to ESP32
    connect_to_esp32()

    # Main Loop
    while True:
        try:
            if mqtt_connected and latest_sensor_flow is not None:
                if latest_sensor_flow > predicted_flow:
                    print(f"Predicted flow:{predicted_flow}")
                    print("Flow too high — Decision: OFF")
                    send_command_to_esp32("OFF")
                else:
                    print(f"Predicted flow:{predicted_flow}")
                    print("Flow OK — Decision: ON")
                    send_command_to_esp32("ON")
                time.sleep(5)  # Check every 5 seconds
        except KeyboardInterrupt:
            print("Exiting...")
            if sock:
                sock.close()
            client.loop_stop()
            client.disconnect()
            break

if __name__ == "__main__":
    main()



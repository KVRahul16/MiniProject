from flask import Flask, request, render_template, send_from_directory
import numpy as np
import pickle
import requests
import os
import cv2
from PIL import Image, ImageDraw, ImageFont

# Load trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Updated soil data mapping (mean values of N, P, K, pH for each soil type)
soil_data = {
    "Acrisols": {"N": 35, "P": 25, "K": 75, "pH": 5.5},
    "Albeluvisols": {"N": 50, "P": 20, "K": 55, "pH": 5.3},
    "Alisols": {"N": 22, "P": 18, "K": 40, "pH": 4.7},
    "Andosols": {"N": 85, "P": 60, "K": 90, "pH": 6.3},
    "Arenosols": {"N": 25, "P": 10, "K": 30, "pH": 5.0},
    "Cambisols": {"N": 60, "P": 45, "K": 80, "pH": 6.8},
    "Chernozems": {"N": 95, "P": 85, "K": 100, "pH": 7.2},
    "Cryosols": {"N": 15, "P": 12, "K": 20, "pH": 5.0},
    "Fluvisols": {"N": 70, "P": 55, "K": 95, "pH": 6.4},
    "Ferralsols": {"N": 28, "P": 15, "K": 35, "pH": 4.5},
    "Gleysols": {"N": 40, "P": 30, "K": 60, "pH": 5.9},
    "Gypsisols": {"N": 28, "P": 14, "K": 35, "pH": 7.5},
    "Histosols": {"N": 45, "P": 32, "K": 65, "pH": 5.2},
    "Kastanozems": {"N": 90, "P": 75, "K": 98, "pH": 7.0},
    "Leptosols": {"N": 30, "P": 18, "K": 50, "pH": 5.5},
    "Lixisols": {"N": 55, "P": 40, "K": 72, "pH": 5.8},
    "Luvisols": {"N": 68, "P": 48, "K": 85, "pH": 6.5},
    "Phaeozems": {"N": 80, "P": 65, "K": 90, "pH": 7.1},
    "Planosols": {"N": 40, "P": 22, "K": 48, "pH": 5.4},
    "Podzols": {"N": 32, "P": 14, "K": 60, "pH": 4.8},
    "Regosols": {"N": 38, "P": 20, "K": 55, "pH": 5.3},
    "Solonchaks": {"N": 20, "P": 12, "K": 30, "pH": 8.2},
    "Solonetz": {"N": 24, "P": 15, "K": 40, "pH": 8.0},
    "Stagnosols": {"N": 35, "P": 22, "K": 50, "pH": 5.5},
    "Vertisols": {"N": 75, "P": 58, "K": 88, "pH": 6.7}
}

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    latitude = float(request.form['Latitude'])
    longitude = float(request.form['Longitude'])
    
    # Fetch soil classification
    soil_type = get_soil_type(latitude, longitude)
    if soil_type not in soil_data:
        return render_template("index.html", result="Soil type not found. Try another location.")
    
    # Fetch soil properties
    N, P, K, pH = soil_data[soil_type].values()

    # Fetch weather details
    temp, humidity, rainfall = get_weather_data(latitude, longitude)
    
    # Prepare input features for model prediction
    feature_list = [N, P, K, pH, temp, humidity, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    # Scale features
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    
    # Make prediction
    prediction = model.predict(sc_mx_features)
    
    # Crop dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    
    crop = crop_dict.get(prediction[0], "Unknown Crop")
    result = f"{crop}"
    
    # Process uploaded image
    if 'file' not in request.files:
        return render_template("index.html", result=result, error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", result=result, error="No selected file.")
    
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    
    # Process image to detect and highlight soil regions and overlay crop name
    processed_filename = os.path.join(PROCESSED_FOLDER, file.filename)
    detect_and_highlight_soil(filename, processed_filename, result)
    
    return render_template("index.html", result=result, processed_image=file.filename)

def detect_and_highlight_soil(image_path, output_path, text):
    """
    Detects soil regions in the image using a simple HSV color segmentation and
    highlights them by drawing bounding boxes. The predicted crop name is also overlaid.
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return

    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for brown color (rough approximation for soil)
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Optionally, apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected soil regions (filter by area)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # adjust threshold as necessary
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Overlay the predicted crop name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Save the processed image
    cv2.imwrite(output_path, img)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

def get_soil_type(lat, lon):
    """Fetches soil classification from ISRIC SoilGrids API."""
    url = "https://rest.isric.org/soilgrids/v2.0/classification/query"
    params = {"lat": lat, "lon": lon, "number_classes": 1}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("wrb_class_name", "Unknown")
    return "Unknown"

def get_weather_data(lat, lon):
    """Fetches weather data from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m"],
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly_data = data.get("hourly", {})
        temp = np.mean(hourly_data.get("temperature_2m", [0]))
        humidity = np.mean(hourly_data.get("relative_humidity_2m", [0]))
        rainfall = np.mean(hourly_data.get("precipitation", [0]))
        return temp, humidity, rainfall
    return 0, 0, 0

if __name__ == "__main__":
    app.run(debug=True)

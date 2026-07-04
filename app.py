import os
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

DATABASE = '/tmp/opticrop.db' if os.environ.get('VERCEL') else 'opticrop.db'

# Crop metadata mapping (Local HD crop images for offline/online reliability)
crop_metadata = {
    'rice': {
        'image': '/static/images/crops/rice.png',
        'desc': 'Rice is a water-intensive cereal crop requiring high humidity, steady rainfall, and clayey-loam soils.'
    },
    'maize': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Maize (Corn) thrives in well-drained fertile soils with moderate rainfall and warm temperatures.'
    },
    'chickpea': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Chickpeas are nutritious pulses that grow well in cool weather and require very low soil moisture.'
    },
    'kidneybeans': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Kidney beans are protein-rich legumes that prefer moderate temperatures and loose, well-aerated soils.'
    },
    'pigeonpeas': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Pigeon peas are drought-tolerant legumes suitable for semi-arid areas with warm temperatures.'
    },
    'mothbeans': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Moth beans are highly drought-resistant legumes grown primarily in hot, arid sandy soils.'
    },
    'mungbean': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Mung beans are quick-growing legumes that adapt well to warm temperatures and dry climates.'
    },
    'blackgram': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Black gram is an essential pulse crop requiring warm climates and moderate rainfall.'
    },
    'lentil': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Lentils are cool-season food legumes that require well-drained, sandy-loam soils and low rainfall.'
    },
    'pomegranate': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Pomegranates are fruit shrubs that thrive in dry climates, preferring well-drained soil and sunny locations.'
    },
    'banana': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Bananas are tropical fruits requiring high humidity, rich organic soil, and heavy water supply.'
    },
    'mango': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Mangoes are tropical fruit trees preferring deep, well-drained soil, hot weather, and distinct dry periods.'
    },
    'grapes': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Grapes require deep, sandy soils, long warm summers, and moderate winter climates for optimal yield.'
    },
    'watermelon': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Watermelons are vine crops that need warm sandy soils, high temperatures, and plenty of sunlight.'
    },
    'muskmelon': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Muskmelons prefer hot, dry weather and fertile, sandy-loam soils with good drainage.'
    },
    'apple': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Apple trees grow best in deep, fertile soils and require distinct cool seasons to develop fruit.'
    },
    'orange': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Oranges are citrus fruits that require sunny, warm climates, well-irrigated soils, and mild winters.'
    },
    'papaya': {
        'image': '/static/images/crops/papaya.png',
        'desc': 'Papayas are tropical fruits that require warm climates, rich sandy soils, and constant soil hydration.'
    },
    'coconut': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Coconuts grow primarily in saline coastal soils, requiring high temperatures and year-round rainfall.'
    },
    'cotton': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Cotton requires deep black soils, warm temperatures, moderate rainfall, and ample sunshine.'
    },
    'jute': {
        'image': '/static/images/crops/default_crop.png',
        'desc': 'Jute requires standing water, warm humid tropical climates, and clayey-alluvial soils.'
    },
    'coffee': {
        'image': '/static/images/crops/coffee.png',
        'desc': 'Coffee thrives in rich volcanic soil, cool mountain temperatures, shaded canopy cover, and moderate rainfall.'
    }
}

# Database functions
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nitrogen REAL,
            phosphorous REAL,
            potassium REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            rainfall REAL,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(n, p, k, temp, hum, ph, rain, prediction):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_history (nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (n, p, k, temp, hum, ph, rain, prediction))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging prediction: {e}")

def get_history(limit=5):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, prediction, timestamp
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []

# Initialize the SQLite Database
init_db()

# Load models and preprocessing pipelines
model = None
scaler = None
label_encoder = None

try:
    with open('best_crop_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Model, scaler, and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/findyourcrop')
def findyourcrop():
    history = get_history(5)
    return render_template('findyourcrop.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    history = get_history(5)
    if model is None or scaler is None or label_encoder is None:
        return render_template('findyourcrop.html', error_text="Error: ML pipelines are not properly loaded.", history=history)
    
    try:
        # Retrieve form parameters explicitly by name
        n_str = request.form.get('nitrogen')
        p_str = request.form.get('phosphorous')
        k_str = request.form.get('potassium')
        temp_str = request.form.get('temperature')
        hum_str = request.form.get('humidity')
        ph_str = request.form.get('ph')
        rain_str = request.form.get('rainfall')

        if not all([n_str, p_str, k_str, temp_str, hum_str, ph_str, rain_str]):
            raise ValueError("All fields are required.")

        n = float(n_str)
        p = float(p_str)
        k = float(k_str)
        temp = float(temp_str)
        hum = float(hum_str)
        ph = float(ph_str)
        rain = float(rain_str)
        
        # Backend Validation Checks
        if not (0 <= n <= 150):
            raise ValueError("Nitrogen (N) must be between 0 and 150 kg/ha.")
        if not (5 <= p <= 150):
            raise ValueError("Phosphorus (P) must be between 5 and 150 kg/ha.")
        if not (5 <= k <= 210):
            raise ValueError("Potassium (K) must be between 5 and 210 kg/ha.")
        if not (0 <= temp <= 50):
            raise ValueError("Temperature must be between 0°C and 50°C.")
        if not (0 <= hum <= 100):
            raise ValueError("Humidity must be between 0% and 100%.")
        if not (3.5 <= ph <= 10.0):
            raise ValueError("Soil pH must be between 3.5 and 10.0.")
        if not (10 <= rain <= 300):
            raise ValueError("Rainfall must be between 10mm and 300mm.")

        # Build pandas DataFrame with the exact features it was fitted on
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=feature_names)
        
        # Preprocess features using StandardScaler
        scaled_features = scaler.transform(features)
        
        # Make ML prediction
        prediction_encoded = model.predict(scaled_features)
        
        # Decode index to human readable crop name
        output_crop = label_encoder.inverse_transform(prediction_encoded)[0]
        prediction_text = 'Best crop for given conditions is {}'.format(output_crop)
        
        # Log to SQLite Database
        log_prediction(n, p, k, temp, hum, ph, rain, output_crop)
        
        # Fetch updated prediction logs
        updated_history = get_history(5)
        
        # Retrieve metadata for predicted crop
        metadata = crop_metadata.get(output_crop.lower(), {
            'image': '/static/images/crops/default_crop.png',
            'desc': 'A suitable crop recommendation based on your soil profile.'
        })
        
        return render_template(
            'findyourcrop.html', 
            prediction_text='Best crop for given conditions is {}'.format(output_crop),
            crop_name=output_crop,
            crop_image=metadata['image'],
            crop_desc=metadata['desc'],
            history=updated_history
        )
        
    except ValueError as e:
        return render_template('findyourcrop.html', error_text='Error: {}'.format(str(e)), history=history)
    except Exception as e:
        return render_template('findyourcrop.html', error_text=f'Error: {str(e)}', history=history)

if __name__ == "__main__":
    app.run(debug=True)

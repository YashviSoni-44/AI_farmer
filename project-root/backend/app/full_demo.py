from flask import Flask, render_template_string, request, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import json
import base64

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# IMPORTANT: Replace with your actual OpenWeatherMap API key
API_KEY = "ef8caec63b44af110ffabc57e1460c52" 

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)

# --- MACHINE LEARNING MODEL SETUP ---
# Global variables for models and encoders
weather_model = None
crop_model = None
crop_encoder = None
weather_model_columns = None

def train_weather_prediction_model():
    """
    Trains the weather prediction model using the provided weather_dataset_3000rows.csv.
    """
    global weather_model, weather_model_columns
    
    # Path is relative to where the app is run from (project-root)
    dataset_path = "project-root/backend/app/weather_dataset_3000rows.csv"
    if not os.path.exists(dataset_path):
        print(f"FATAL ERROR: Weather dataset '{dataset_path}' not found.")
        print("Please ensure the dataset file is in the correct directory.")
        exit()

    df = pd.read_csv(dataset_path)

    df["rain_probability"] = df["rain_probability"].str.replace("%", "").astype(float)
    target_columns = ["rainfall_mm", "rain_duration_hours", "rain_probability", "temp_avg_c"]
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    X = pd.get_dummies(X, drop_first=True)
    weather_model_columns = X.columns

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)
    
    weather_model = model
    print("Weather prediction model trained successfully from CSV.")

def train_crop_recommendation_model():
    """
    Trains the crop recommendation model using diverse_crop_suitability_balanced_with_soil.csv.
    """
    global crop_model, crop_encoder
    
    dataset_path = "project-root/backend/app/diverse_crop_suitability_balanced_with_soil.csv"
    if not os.path.exists(dataset_path):
        print(f"FATAL ERROR: Crop dataset '{dataset_path}' not found.")
        print("Please ensure the dataset file is in the correct directory.")
        exit()
        
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["recommended_crop"])
    y = df["recommended_crop"]
    le_crop = LabelEncoder()
    y_encoded = le_crop.fit_transform(y)
    X = pd.get_dummies(X, columns=['soil_type'], drop_first=False)
    X_train_cols = X.columns
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    crop_model = (model, X_train_cols)
    crop_encoder = le_crop
    print("Crop recommendation model trained successfully from CSV.")


# --- DATA FETCHING & PROCESSING ---
def get_weather_and_rainfall_prediction(city_name):
    """
    Fetches weather data from OpenWeatherMap and predicts rainfall using the trained model.
    """
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
    geo_response = requests.get(geo_url)
    if geo_response.status_code != 200 or not geo_response.json():
        raise ValueError("Could not find coordinates for the city.")
    
    coords = geo_response.json()[0]
    lat, lon = coords['lat'], coords['lon']
    state = coords.get('state', 'N/A')

    forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    forecast_response = requests.get(forecast_url)
    if forecast_response.status_code != 200:
        raise ValueError("Could not fetch weather forecast.")
    
    forecast_data = forecast_response.json()
    
    data = [
        {
            'temp_high_C': round(entry['main']['temp_max'] - 273.15, 1),
            'temp_low_C': round(entry['main']['temp_min'] - 273.15, 1),
            'humidity_%': entry['main']['humidity'],
            'wind_speed_mps': entry['wind']['speed'],
        }
        for entry in forecast_data['list']
    ]
    
    df = pd.DataFrame(data)
    mean_values = df.mean()

    temp_high = round(mean_values["temp_high_C"], 1)
    temp_low = round(mean_values["temp_low_C"], 1)
    humidity = round(mean_values["humidity_%"], 1)
    wind_speed = round(mean_values["wind_speed_mps"] * 3.6, 1)

    month_map = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
    month_str = forecast_data['list'][0]['dt_txt'].split('-')[1]
    month = f'{month_map[month_str]}-25'

    weather_input_data = pd.DataFrame([{
        "state": state, "district": city_name, "month": month,
        "temp_high_c": temp_high, "temp_low_c": temp_low,
        "humidity_percent": humidity, "sunlight_hours": 8.0,
        "wind_speed_kmph": wind_speed, "wind_direction": "nW"
    }])
    
    weather_input_encoded = pd.get_dummies(weather_input_data)
    weather_input_aligned = weather_input_encoded.reindex(columns=weather_model_columns, fill_value=0)

    weather_prediction = weather_model.predict(weather_input_aligned)[0]
    
    return {
        "location": {"city": city_name.title(), "state": state},
        "forecast": {
            "temp_high": temp_high, "temp_low": temp_low,
            "humidity": humidity, "wind_speed": wind_speed
        },
        "rainfall_prediction": {
            "rainfall_mm": max(0, round(weather_prediction[0], 1)),
            "rain_duration_hours": max(0, round(weather_prediction[1], 1)),
            "rain_probability": min(100, max(0, round(weather_prediction[2], 1))),
            "temp_avg": round(weather_prediction[3], 1)
        }
    }

def get_crop_recommendation(prediction_data):
    """
    Recommends a crop based on weather data and soil type using the trained model.
    """
    crop_input_data = pd.DataFrame([prediction_data])
    crop_input_encoded = pd.get_dummies(crop_input_data, columns=['soil_type'], drop_first=False)
    model, required_cols = crop_model
    crop_input_aligned = crop_input_encoded.reindex(columns=required_cols, fill_value=0)
    predicted_crop_encoded = model.predict(crop_input_aligned)[0]
    predicted_crop = crop_encoder.inverse_transform([predicted_crop_encoded])[0]
    return {"crop": predicted_crop}

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    city = request.json.get('city')
    if not city:
        return jsonify({"error": "City name is required."}), 400
    try:
        return jsonify(get_weather_and_rainfall_prediction(city))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.json
    try:
        return jsonify(get_crop_recommendation(data))
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handles requests to the AI chatbot."""
    data = request.json
    user_message = data.get('message')
    language = data.get('language', 'English')
    image_data = data.get('image')

    # Construct the payload for the Gemini API
    system_prompt = f"""You are 'Agri-Mitra', an expert AI assistant for Indian farmers. Your goal is to provide clear, concise, and actionable advice. Respond exclusively in {language}.
    - When asked about government schemes, use your search tool to find the most current information for India, and specifically for states like Gujarat or others if mentioned. Explain eligibility, benefits, and application processes simply.
    - For plant diseases, analyze the user's image, identify the disease, and suggest both organic and chemical treatment options.
    - For general farming advice (like planting times, watering, fertilizers), be practical and consider the context of Indian agriculture.
    - For community questions, answer them based on your broad knowledge.
    - Keep your answers easy to understand for all farmers."""
    
    payload_contents = [{'role': 'user', 'parts': []}]
    
    if image_data:
        # Multimodal request for disease detection
        image_bytes = base64.b64decode(image_data.split(',')[1])
        payload_contents[0]['parts'].append({
            'inlineData': {
                'mimeType': 'image/jpeg',
                'data': base64.b64encode(image_bytes).decode('utf-8')
            }
        })
        # Add a specific prompt for disease analysis if user message is generic
        if not user_message or user_message.strip() == "":
            user_message = "Please analyze this image and tell me if my plant is sick. What disease is it and how can I treat it?"

    payload_contents[0]['parts'].append({'text': user_message})

    payload = {
        'contents': payload_contents,
        'tools': [{'google_search': {}}],
        'systemInstruction': {
            'parts': [{'text': system_prompt}]
        },
    }
    
    try:
        # Use an empty API key, letting the environment provide it
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
        
        response = requests.post(
            api_url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        
        result = response.json()
        bot_response = result['candidates'][0]['content']['parts'][0]['text']
        
        return jsonify({'reply': bot_response})
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'error': 'Failed to get a response from the AI assistant.'}), 500

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriCast - Weather & Crop Advisor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f0f4f8; }
        .card { background-color: white; border-radius: 1rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05); transition: all 0.3s ease; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04); }
        .icon { width: 50px; height: 50px; }
        .hidden-section { display: none; opacity: 0; }
        .visible-section { display: block; opacity: 1; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        /* Chatbot Styles */
        #chat-widget { position: fixed; bottom: 2rem; right: 2rem; z-index: 1000; }
        #chat-container { width: 400px; height: 550px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); transition: all 0.3s ease-in-out; transform-origin: bottom right; }
        #chat-messages { scroll-behavior: smooth; }
        .user-msg { background-color: #3b82f6; color: white; align-self: flex-end; }
        .bot-msg { background-color: #e5e7eb; color: #1f2937; align-self: flex-start; }
        .mic-active { color: #ef4444; }
    </style>
</head>
<body class="antialiased text-gray-800">
    <div class="min-h-screen flex flex-col items-center justify-center p-4 bg-gradient-to-br from-blue-100 to-green-100">
        <div class="w-full max-w-4xl mx-auto">
            <header class="text-center mb-8">
                <h1 class="text-4xl md:text-5xl font-bold text-gray-900">AgriCast</h1>
                <p class="text-lg text-gray-600 mt-2">Your AI-Powered Farming Advisor</p>
            </header>

            <main>
                <!-- Existing content -->
                <div class="card p-6 md:p-8 mb-8">
                    <form id="weather-form" class="flex flex-col sm:flex-row items-center gap-4">
                        <input type="text" id="city-input" name="city" placeholder="Enter your district or city name..." class="w-full px-4 py-3 text-lg border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 transition duration-300" required>
                        <button type="submit" class="w-full sm:w-auto bg-blue-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-300">Get Forecast</button>
                    </form>
                </div>
                <div id="loader" class="hidden justify-center items-center my-8"><div class="loader"></div><p class="ml-4 text-gray-600">Analyzing...</p></div>
                <div id="error-message" class="hidden text-center text-red-600 bg-red-100 p-4 rounded-lg"></div>
                <div id="weather-results-container" class="hidden-section grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div class="card p-6 col-span-1 md:col-span-1 flex flex-col">
                        <h2 class="text-xl font-semibold mb-4 text-gray-700">5-Day Avg Forecast</h2>
                        <div class="flex items-center space-x-4 mb-4"><svg class="icon text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" /></svg><div><p class="text-sm text-gray-500">Avg. High / Low</p><p id="temp" class="text-2xl font-bold"></p></div></div>
                        <div class="flex items-center space-x-4 mb-4"><svg class="icon text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.362-3.797z" /></svg><div><p class="text-sm text-gray-500">Avg. Humidity</p><p id="humidity" class="text-2xl font-bold"></p></div></div>
                        <div class="flex items-center space-x-4"><svg class="icon text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" /></svg><div><p class="text-sm text-gray-500">Avg. Wind Speed</p><p id="wind" class="text-2xl font-bold"></p></div></div>
                    </div>
                    <div class="card p-6 col-span-1 md:col-span-2 flex flex-col">
                        <h2 class="text-xl font-semibold mb-4 text-gray-700">Rainfall Prediction</h2>
                        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
                            <div><p class="text-sm text-gray-500">Probability</p><p id="rain-prob" class="text-3xl font-bold text-blue-600"></p></div>
                            <div><p class="text-sm text-gray-500">Amount (mm)</p><p id="rain-mm" class="text-3xl font-bold text-blue-600"></p></div>
                            <div><p class="text-sm text-gray-500">Duration (hrs)</p><p id="rain-hours" class="text-3xl font-bold text-blue-600"></p></div>
                        </div>
                    </div>
                </div>
                <div id="crop-prediction-section" class="hidden-section card p-6 md:p-8">
                    <h2 class="text-2xl font-semibold mb-4 text-gray-800">Get Crop Recommendation</h2>
                    <form id="crop-form" class="flex flex-col sm:flex-row items-center gap-4">
                        <select id="soil-type-select" class="w-full px-4 py-3 text-lg border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 transition duration-300"><option value="Loamy">Loamy Soil</option><option value="Sandy">Sandy Soil</option><option value="Clay">Clay Soil</option><option value="Black">Black Soil</option><option value="Red">Red Soil</option></select>
                        <button type="submit" class="w-full sm:w-auto bg-green-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-4 focus:ring-green-300">Predict Crop</button>
                    </form>
                </div>
                <div id="crop-loader" class="hidden justify-center items-center my-8"><div class="loader"></div><p class="ml-4 text-gray-600">Recommending crop...</p></div>
                <div id="crop-result-card" class="hidden-section card p-6 mt-6 bg-gradient-to-r from-green-500 to-emerald-600 text-white"><div class="flex flex-col md:flex-row items-center text-center md:text-left"><div class="mb-4 md:mb-0 md:mr-6"><p class="text-lg">Based on the forecast for <strong id="location" class="font-bold"></strong> and your selected soil, the recommended crop is:</p><p id="crop-name" class="text-5xl font-extrabold tracking-tight mt-2"></p></div><div class="md:ml-auto"><img id="crop-image" src="" alt="Recommended Crop" class="w-32 h-32 object-cover rounded-full shadow-lg border-4 border-white/50"></div></div></div>
            </main>
        </div>
    </div>

    <!-- AI Chatbot Widget -->
    <div id="chat-widget">
        <div id="chat-container" class="hidden-section card fixed bottom-24 right-8 w-[400px] h-[550px] flex-col rounded-2xl overflow-hidden">
            <!-- Header -->
            <div class="bg-blue-600 text-white p-4 flex justify-between items-center">
                <div>
                    <h3 class="font-bold text-lg">Agri-Mitra Assistant</h3>
                    <div class="flex items-center gap-x-2 mt-1">
                        <div class="w-2 h-2 bg-green-400 rounded-full"></div>
                        <p class="text-xs">Online</p>
                    </div>
                </div>
                <select id="language-select" class="bg-blue-500 text-white text-sm rounded-md border-0 focus:ring-2 focus:ring-white">
                    <option value="en-IN">English</option>
                    <option value="hi-IN">हिन्दी</option>
                    <option value="gu-IN">ગુજરાતી</option>
                </select>
            </div>
            <!-- Messages -->
            <div id="chat-messages" class="flex-1 p-4 overflow-y-auto bg-gray-100 flex flex-col gap-3">
                <div class="bot-msg p-3 rounded-lg max-w-xs">Hello! I am Agri-Mitra. How can I assist you with your farming questions today?</div>
            </div>
            <!-- Input -->
            <div class="p-4 bg-white border-t">
                <div id="image-preview-container" class="hidden items-center mb-2">
                    <img id="image-preview" class="w-16 h-16 rounded-md object-cover" />
                    <button id="remove-image-btn" class="ml-2 text-red-500 font-bold">X</button>
                </div>
                <div class="flex items-center gap-2">
                    <input type="text" id="chat-input" placeholder="Ask a question..." class="w-full px-4 py-2 border rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <input type="file" id="image-upload" class="hidden" accept="image/*">
                    <button id="image-upload-btn" class="p-2 rounded-full hover:bg-gray-200"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg></button>
                    <button id="mic-btn" class="p-2 rounded-full hover:bg-gray-200"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg></button>
                </div>
            </div>
        </div>
        <button id="chat-toggle" class="bg-blue-600 text-white w-16 h-16 rounded-full flex items-center justify-center shadow-lg hover:bg-blue-700 transition-transform transform hover:scale-110">
            <svg id="chat-icon-open" xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 21 1.65-3.8a9 9 0 1 1 3.4 2.9l-5.05 .9"></path><path d="M9 10a.5.5 0 0 0 1 0V4a.5.5 0 0 0-1 0v6Z"></path><path d="M12 15a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-1 0v2Z"></path><path d="M15 12a.5.5 0 0 0 1 0V9a.5.5 0 0 0-1 0v3Z"></path></svg>
            <svg id="chat-icon-close" class="hidden" xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        </button>
    </div>

    <script>
        // --- Existing Weather/Crop Script ---
        const weatherForm = document.getElementById('weather-form');
        const cityInput = document.getElementById('city-input');
        const loader = document.getElementById('loader');
        const errorMessage = document.getElementById('error-message');
        const weatherResultsContainer = document.getElementById('weather-results-container');
        const cropPredictionSection = document.getElementById('crop-prediction-section');
        const cropForm = document.getElementById('crop-form');
        const soilTypeSelect = document.getElementById('soil-type-select');
        const cropLoader = document.getElementById('crop-loader');
        const cropResultCard = document.getElementById('crop-result-card');
        let currentWeatherData = null;
        const cropImages={'Wheat':'https://placehold.co/200x200/F4A460/FFFFFF?text=Wheat','Corn':'https://placehold.co/200x200/FBEC5D/000000?text=Corn','Rice':'https://placehold.co/200x200/F5F5DC/000000?text=Rice','Sugarcane':'https://placehold.co/200x200/90EE90/000000?text=Sugarcane','Cotton':'https://placehold.co/200x200/FFFAFA/000000?text=Cotton','Bajra':'https://placehold.co/200x200/BDB76B/FFFFFF?text=Bajra','Groundnut':'https://placehold.co/200x200/D2B48C/FFFFFF?text=Groundnut','Jowar':'https://placehold.co/200x200/DAA520/FFFFFF?text=Jowar','Maize':'https://placehold.co/200x200/FFD700/000000?text=Maize','Soyabean':'https://placehold.co/200x200/ADFF2F/000000?text=Soyabean','Default':'https://placehold.co/200x200/CCCCCC/FFFFFF?text=Crop'};
        weatherForm.addEventListener('submit',async e=>{e.preventDefault();const t=cityInput.value.trim();if(!t)return;hideAllResults();errorMessage.style.display='none';loader.style.display='flex';try{const e=await fetch('/predict_weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({city:t})});const o=await e.json();if(!e.ok)throw new Error(o.error||'Something went wrong');currentWeatherData=o;updateWeatherUI(o)}catch(e){errorMessage.textContent=`Error: ${e.message}`;errorMessage.style.display='block'}finally{loader.style.display='none'}});
        cropForm.addEventListener('submit',async e=>{e.preventDefault();if(!currentWeatherData)return;cropResultCard.classList.remove('visible-section');cropResultCard.classList.add('hidden-section');errorMessage.style.display='none';cropLoader.style.display='flex';const t={soil_type:soilTypeSelect.value,rainfall_mm:currentWeatherData.rainfall_prediction.rainfall_mm,temp_high_c:currentWeatherData.forecast.temp_high,temp_low_c:currentWeatherData.forecast.temp_low,temp_avg_c:currentWeatherData.rainfall_prediction.temp_avg,humidity_percent:currentWeatherData.forecast.humidity,wind_speed_kmph:currentWeatherData.forecast.wind_speed,sunlight_hours:8};try{const e=await fetch('/predict_crop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(t)});const o=await e.json();if(!e.ok)throw new Error(o.error||'Something went wrong');updateCropUI(o)}catch(e){errorMessage.textContent=`Error: ${e.message}`;errorMessage.style.display='block'}finally{cropLoader.style.display='none'}});
        function hideAllResults(){weatherResultsContainer.classList.remove('visible-section');weatherResultsContainer.classList.add('hidden-section');cropPredictionSection.classList.remove('visible-section');cropPredictionSection.classList.add('hidden-section');cropResultCard.classList.remove('visible-section');cropResultCard.classList.add('hidden-section')}
        function updateWeatherUI(e){document.getElementById('temp').textContent=`${e.forecast.temp_high}°C / ${e.forecast.temp_low}°C`;document.getElementById('humidity').textContent=`${e.forecast.humidity}%`;document.getElementById('wind').textContent=`${e.forecast.wind_speed} km/h`;document.getElementById('rain-prob').textContent=`${e.rainfall_prediction.rain_probability}%`;document.getElementById('rain-mm').textContent=`${e.rainfall_prediction.rainfall_mm}`;document.getElementById('rain-hours').textContent=`${e.rainfall_prediction.rain_duration_hours}`;weatherResultsContainer.classList.remove('hidden-section');cropPredictionSection.classList.remove('hidden-section');setTimeout(()=>{weatherResultsContainer.classList.add('visible-section');cropPredictionSection.classList.add('visible-section')},10)}
        function updateCropUI(e){document.getElementById('location').textContent=`${currentWeatherData.location.city}, ${currentWeatherData.location.state}`;const t=e.crop;document.getElementById('crop-name').textContent=t;document.getElementById('crop-image').src=cropImages[t]||cropImages['Default'];document.getElementById('crop-image').alt=t;cropResultCard.classList.remove('hidden-section');setTimeout(()=>{cropResultCard.classList.add('visible-section')},10)}
        
        // --- New AI Chatbot Script ---
        const chatContainer = document.getElementById('chat-container');
        const chatToggle = document.getElementById('chat-toggle');
        const chatIconOpen = document.getElementById('chat-icon-open');
        const chatIconClose = document.getElementById('chat-icon-close');
        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const micBtn = document.getElementById('mic-btn');
        const langSelect = document.getElementById('language-select');
        const imageUploadBtn = document.getElementById('image-upload-btn');
        const imageUploadInput = document.getElementById('image-upload');
        const imagePreviewContainer = document.getElementById('image-preview-container');
        const imagePreview = document.getElementById('image-preview');
        const removeImageBtn = document.getElementById('remove-image-btn');
        
        let conversationHistory = [];
        let isChatOpen = false;
        let attachedImage = null;

        // --- Voice Recognition Setup ---
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition;
        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.lang = langSelect.value;
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            recognition.onresult = (event) => {
                const speechResult = event.results[0][0].transcript;
                chatInput.value = speechResult;
                sendMessage();
            };
            recognition.onaudiostart = () => micBtn.classList.add('mic-active');
            recognition.onaudioend = () => micBtn.classList.remove('mic-active');
        } else {
            micBtn.style.display = 'none'; // Hide if not supported
        }

        // --- Event Listeners ---
        chatToggle.addEventListener('click', () => {
            isChatOpen = !isChatOpen;
            chatContainer.classList.toggle('hidden-section');
            chatIconOpen.classList.toggle('hidden');
            chatIconClose.classList.toggle('hidden');
            if(isChatOpen) {
                chatContainer.style.transform = 'scale(1)';
            } else {
                chatContainer.style.transform = 'scale(0)';
            }
        });

        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        micBtn.addEventListener('click', () => {
            if (recognition) {
                recognition.lang = langSelect.value;
                recognition.start();
            }
        });
        
        imageUploadBtn.addEventListener('click', () => imageUploadInput.click());
        
        imageUploadInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    attachedImage = e.target.result;
                    imagePreview.src = attachedImage;
                    imagePreviewContainer.classList.remove('hidden');
                    imagePreviewContainer.classList.add('flex');
                };
                reader.readAsDataURL(file);
            }
        });

        removeImageBtn.addEventListener('click', () => {
            attachedImage = null;
            imageUploadInput.value = ''; // Reset file input
            imagePreviewContainer.classList.add('hidden');
        });

        function addMessageToUI(message, sender) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `p-3 rounded-lg max-w-xs ${sender === 'user' ? 'user-msg' : 'bot-msg'}`;
            msgDiv.textContent = message;
            chatMessages.appendChild(msgDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function sendMessage() {
            const message = chatInput.value.trim();
            if (!message && !attachedImage) return;

            addMessageToUI(message, 'user');
            chatInput.value = '';
            
            // Show typing indicator
            const typingIndicator = document.createElement('div');
            typingIndicator.id = 'typing-indicator';
            typingIndicator.className = 'bot-msg p-3 rounded-lg max-w-xs';
            typingIndicator.textContent = '...';
            chatMessages.appendChild(typingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        language: langSelect.options[langSelect.selectedIndex].text,
                        image: attachedImage
                    })
                });
                const data = await response.json();
                
                chatMessages.removeChild(typingIndicator); // Remove typing indicator
                
                if (data.error) {
                    addMessageToUI(`Error: ${data.error}`, 'bot');
                } else {
                    addMessageToUI(data.reply, 'bot');
                    speak(data.reply, langSelect.value);
                }

            } catch (error) {
                chatMessages.removeChild(typingIndicator);
                addMessageToUI('Sorry, I am having trouble connecting. Please try again.', 'bot');
            }
            
            // Clear image after sending
            if (attachedImage) {
                removeImageBtn.click();
            }
        }

        function speak(text, lang) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = lang;
            speechSynthesis.speak(utterance);
        }
    </script>
</body>
</html>
"""

# --- RUN THE APP ---
if __name__ == '__main__':
    # Train (or load) models on startup
    train_weather_prediction_model()
    train_crop_recommendation_model()
    app.run(host='0.0.0.0', port=5001, debug=False)


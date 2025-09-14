# from flask import Flask, render_template_string, request, jsonify
# import requests
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.preprocessing import LabelEncoder
# import warnings

# # Suppress warnings for a cleaner output
# warnings.filterwarnings('ignore')

# # --- CONFIGURATION ---
# # IMPORTANT: Replace with your actual OpenWeatherMap API key
# API_KEY = "ef8caec63b44af110ffabc57e1460c52" 

# # --- INITIALIZE FLASK APP ---
# app = Flask(__name__)

# # --- MACHINE LEARNING MODEL SETUP ---
# # Global variables for models and encoders
# weather_model = None
# crop_model = None
# crop_encoder = None
# weather_model_columns = None

# def train_weather_prediction_model():
#     """
#     Trains the weather prediction model using a sample dataset.
#     In a real application, you would load a pre-trained model.
#     """
#     global weather_model, weather_model_columns
    
#     # Create a dummy dataset resembling the expected input
#     data = {
#         'state': ['StateA']*50 + ['StateB']*50,
#         'district': ['CityA']*25 + ['CityB']*25 + ['CityC']*25 + ['CityD']*25,
#         'month': ['Jan-25', 'Feb-25', 'Mar-25', 'Apr-25'] * 25,
#         'temp_high_c': np.random.uniform(25, 40, 100),
#         'temp_low_c': np.random.uniform(10, 25, 100),
#         'humidity_percent': np.random.uniform(30, 90, 100),
#         'sunlight_hours': [8] * 100,
#         'wind_speed_kmph': np.random.uniform(5, 20, 100),
#         'wind_direction': ['nW', 'sE'] * 50,
#         'rainfall_mm': np.random.uniform(0, 200, 100),
#         'rain_duration_hours': np.random.uniform(0, 12, 100),
#         'rain_probability': np.random.uniform(0, 100, 100),
#         'temp_avg_c': np.random.uniform(15, 35, 100)
#     }
#     df = pd.DataFrame(data)

#     target_columns = ["rainfall_mm", "rain_duration_hours", "rain_probability", "temp_avg_c"]
#     X = df.drop(columns=target_columns)
#     y = df[target_columns]

#     X = pd.get_dummies(X, drop_first=True)
#     weather_model_columns = X.columns

#     # Using a simple model for demonstration
#     model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
#     model.fit(X, y)
    
#     weather_model = model
#     print("Weather prediction model trained successfully.")

# def train_crop_recommendation_model():
#     """
#     Trains the crop recommendation model using a sample dataset.
#     In a real application, you would load a pre-trained model.
#     """
#     global crop_model, crop_encoder
    
#     # Create a dummy dataset
#     data = {
#         'soil_type': ['Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy'] * 20,
#         'rainfall_mm': np.random.uniform(50, 250, 100),
#         'temp_high_c': np.random.uniform(25, 42, 100),
#         'temp_low_c': np.random.uniform(10, 26, 100),
#         'temp_avg_c': np.random.uniform(18, 38, 100),
#         'humidity_percent': np.random.uniform(30, 95, 100),
#         'sunlight_hours': [8] * 100,
#         'wind_speed_kmph': np.random.uniform(5, 25, 100),
#         'recommended_crop': ['Wheat', 'Corn', 'Rice', 'Sugarcane', 'Cotton'] * 20
#     }
#     df = pd.DataFrame(data)
    
#     X = df.drop(columns=["recommended_crop"])
#     y = df["recommended_crop"]

#     le_crop = LabelEncoder()
#     y_encoded = le_crop.fit_transform(y)
    
#     X = pd.get_dummies(X, columns=['soil_type'], drop_first=True)

#     # Align columns for prediction time
#     X_train_cols = X.columns
    
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y_encoded)

#     crop_model = (model, X_train_cols) # Save model and columns
#     crop_encoder = le_crop
#     print("Crop recommendation model trained successfully.")


# # --- DATA FETCHING & PROCESSING ---
# def get_weather_data_and_predict(city_name):
#     """
#     Fetches weather data, processes it, and runs predictions.
#     """
#     # 1. Get Coordinates
#     geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
#     geo_response = requests.get(geo_url)
#     if geo_response.status_code != 200 or not geo_response.json():
#         raise ValueError("Could not find coordinates for the city.")
    
#     coords = geo_response.json()[0]
#     lat, lon = coords['lat'], coords['lon']
#     state = coords.get('state', 'N/A')

#     # 2. Get 5-Day Forecast
#     forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
#     forecast_response = requests.get(forecast_url)
#     if forecast_response.status_code != 200:
#         raise ValueError("Could not fetch weather forecast.")
    
#     forecast_data = forecast_response.json()

#     # 3. Process Forecast Data
#     data = []
#     for entry in forecast_data['list']:
#         data.append({
#             'temp_high_C': round(entry['main']['temp_max'] - 273.15, 1),
#             'temp_low_C': round(entry['main']['temp_min'] - 273.15, 1),
#             'humidity_%': entry['main']['humidity'],
#             'wind_speed_mps': entry['wind']['speed']
#         })
    
#     df = pd.DataFrame(data)
#     mean_values = df.mean()

#     # 4. Prepare Data for Weather Prediction Model
#     temp_high = round(mean_values["temp_high_C"], 1)
#     temp_low = round(mean_values["temp_low_C"], 1)
#     humidity = round(mean_values["humidity_%"], 1)
#     wind_speed = round(mean_values["wind_speed_mps"] * 3.6, 1) # m/s to km/h

#     month_map = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
#     month_str = forecast_data['list'][0]['dt_txt'].split('-')[1]
#     month = f'{month_map[month_str]}-25'

#     weather_input_data = pd.DataFrame([{
#         "state": state,
#         "district": city_name,
#         "month": month,
#         "temp_high_c": temp_high,
#         "temp_low_c": temp_low,
#         "humidity_percent": humidity,
#         "sunlight_hours": 8.0,
#         "wind_speed_kmph": wind_speed,
#         "wind_direction": "nW"
#     }])
    
#     weather_input_encoded = pd.get_dummies(weather_input_data)
#     weather_input_aligned = weather_input_encoded.reindex(columns=weather_model_columns, fill_value=0)

#     # 5. Predict Weather
#     weather_prediction = weather_model.predict(weather_input_aligned)[0]
#     rainfall_mm = max(0, round(weather_prediction[0], 1))
#     rain_duration = max(0, round(weather_prediction[1], 1))
#     rain_prob = min(100, max(0, round(weather_prediction[2], 1)))
#     temp_avg = round(weather_prediction[3], 1)
    
#     # 6. Prepare Data for Crop Prediction Model
#     # For this example, we assume a "Loamy" soil type. You could add a dropdown for this.
#     crop_input_data = pd.DataFrame([{
#         "soil_type": "Loamy",
#         "rainfall_mm": rainfall_mm,
#         "temp_high_c": temp_high,
#         "temp_low_c": temp_low,
#         "temp_avg_c": temp_avg,
#         "humidity_percent": humidity,
#         "sunlight_hours": 8.0,
#         "wind_speed_kmph": wind_speed
#     }])
    
#     crop_input_encoded = pd.get_dummies(crop_input_data, columns=['soil_type'], drop_first=True)
    
#     model, required_cols = crop_model
#     crop_input_aligned = crop_input_encoded.reindex(columns=required_cols, fill_value=0)

#     # 7. Predict Crop
#     predicted_crop_encoded = model.predict(crop_input_aligned)[0]
#     predicted_crop = crop_encoder.inverse_transform([predicted_crop_encoded])[0]

#     # Compile all results
#     results = {
#         "location": {"city": city_name.title(), "state": state},
#         "forecast": {
#             "temp_high": temp_high,
#             "temp_low": temp_low,
#             "humidity": humidity,
#             "wind_speed": wind_speed
#         },
#         "rainfall_prediction": {
#             "rainfall_mm": rainfall_mm,
#             "rain_duration_hours": rain_duration,
#             "rain_probability": rain_prob
#         },
#         "crop_recommendation": {
#             "crop": predicted_crop,
#             "soil_type": "Loamy"
#         }
#     }
    
#     return results

# # --- FLASK ROUTES ---
# @app.route('/')
# def index():
#     """Renders the main HTML page."""
#     return render_template_string(HTML_TEMPLATE)

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handles the prediction request from the frontend."""
#     city = request.json.get('city')
#     if not city:
#         return jsonify({"error": "City name is required."}), 400
    
#     try:
#         prediction_results = get_weather_data_and_predict(city)
#         return jsonify(prediction_results)
#     except ValueError as e:
#         return jsonify({"error": str(e)}), 404
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return jsonify({"error": "An internal error occurred. Please try again later."}), 500

# # --- HTML TEMPLATE ---
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>AgriCast - Weather & Crop Advisor</title>
#     <script src="https://cdn.tailwindcss.com"></script>
#     <link rel="preconnect" href="https://fonts.googleapis.com">
#     <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
#     <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
#     <style>
#         body {
#             font-family: 'Inter', sans-serif;
#             background-color: #f0f4f8;
#         }
#         .card {
#             background-color: white;
#             border-radius: 1rem;
#             box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
#             transition: transform 0.3s ease, box-shadow 0.3s ease;
#         }
#         .card:hover {
#             transform: translateY(-5px);
#             box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
#         }
#         .icon {
#             width: 50px;
#             height: 50px;
#         }
#         #results-container {
#             display: none;
#             opacity: 0;
#             transition: opacity 0.5s ease-in-out;
#         }
#         .fade-in {
#             opacity: 1 !important;
#         }
#         .loader {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #3498db;
#             border-radius: 50%;
#             width: 40px;
#             height: 40px;
#             animation: spin 1s linear infinite;
#         }
#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }
#     </style>
# </head>
# <body class="antialiased text-gray-800">
#     <div class="min-h-screen flex flex-col items-center justify-center p-4 bg-gradient-to-br from-blue-100 to-green-100">
#         <div class="w-full max-w-4xl mx-auto">
#             <header class="text-center mb-8">
#                 <h1 class="text-4xl md:text-5xl font-bold text-gray-900">AgriCast</h1>
#                 <p class="text-lg text-gray-600 mt-2">Your AI-Powered Farming Advisor</p>
#             </header>

#             <main>
#                 <div class="card p-6 md:p-8 mb-8">
#                     <form id="prediction-form" class="flex flex-col sm:flex-row items-center gap-4">
#                         <input type="text" id="city-input" name="city" placeholder="Enter your district or city name..." class="w-full px-4 py-3 text-lg border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-300" required>
#                         <button type="submit" class="w-full sm:w-auto bg-blue-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-300 transition duration-300">
#                             Get Prediction
#                         </button>
#                     </form>
#                 </div>

#                 <div id="loader" class="hidden justify-center items-center my-8">
#                     <div class="loader"></div>
#                     <p class="ml-4 text-gray-600">Fetching data and running AI models...</p>
#                 </div>

#                 <div id="error-message" class="hidden text-center text-red-600 bg-red-100 p-4 rounded-lg"></div>

#                 <div id="results-container" class="grid grid-cols-1 md:grid-cols-3 gap-6">
#                     <!-- Weather Forecast Card -->
#                     <div class="card p-6 col-span-1 md:col-span-1 flex flex-col">
#                         <h2 class="text-xl font-semibold mb-4 text-gray-700">5-Day Average Forecast</h2>
#                         <div class="flex items-center space-x-4 mb-4">
#                            <svg class="icon text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" /></svg>
#                            <div>
#                                <p class="text-sm text-gray-500">Avg. High / Low</p>
#                                <p id="temp" class="text-2xl font-bold"></p>
#                            </div>
#                         </div>
#                         <div class="flex items-center space-x-4 mb-4">
#                             <svg class="icon text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.362-3.797z" /></svg>
#                             <div>
#                                 <p class="text-sm text-gray-500">Avg. Humidity</p>
#                                 <p id="humidity" class="text-2xl font-bold"></p>
#                             </div>
#                         </div>
#                         <div class="flex items-center space-x-4">
#                              <svg class="icon text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" /></svg>
#                              <div>
#                                 <p class="text-sm text-gray-500">Avg. Wind Speed</p>
#                                 <p id="wind" class="text-2xl font-bold"></p>
#                             </div>
#                         </div>
#                     </div>
                    
#                     <!-- Rainfall Prediction Card -->
#                     <div class="card p-6 col-span-1 md:col-span-2 flex flex-col">
#                         <h2 class="text-xl font-semibold mb-4 text-gray-700">Rainfall Prediction</h2>
#                         <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
#                             <div>
#                                 <p class="text-sm text-gray-500">Probability</p>
#                                 <p id="rain-prob" class="text-3xl font-bold text-blue-600"></p>
#                             </div>
#                             <div>
#                                 <p class="text-sm text-gray-500">Amount (mm)</p>
#                                 <p id="rain-mm" class="text-3xl font-bold text-blue-600"></p>
#                             </div>
#                             <div>
#                                 <p class="text-sm text-gray-500">Duration (hrs)</p>
#                                 <p id="rain-hours" class="text-3xl font-bold text-blue-600"></p>
#                             </div>
#                         </div>
#                     </div>

#                     <!-- Crop Recommendation Card -->
#                     <div class="card p-6 col-span-1 md:col-span-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white">
#                         <h2 class="text-2xl font-semibold mb-4">AI Crop Recommendation</h2>
#                         <div class="flex flex-col md:flex-row items-center text-center md:text-left">
#                             <div class="mb-4 md:mb-0 md:mr-6">
#                                 <p class="text-lg">Based on the forecast for <strong id="location" class="font-bold"></strong> and assuming <strong class="font-bold">Loamy Soil</strong>, the recommended crop is:</p>
#                                 <p id="crop-name" class="text-5xl font-extrabold tracking-tight mt-2"></p>
#                             </div>
#                             <div class="md:ml-auto">
#                                 <img id="crop-image" src="" alt="Recommended Crop" class="w-32 h-32 object-cover rounded-full shadow-lg border-4 border-white/50">
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#             </main>
#         </div>
#     </div>

#     <script>
#         const form = document.getElementById('prediction-form');
#         const cityInput = document.getElementById('city-input');
#         const loader = document.getElementById('loader');
#         const errorMessage = document.getElementById('error-message');
#         const resultsContainer = document.getElementById('results-container');

#         const cropImages = {
#             'Wheat': 'https://placehold.co/200x200/F4A460/FFFFFF?text=Wheat',
#             'Corn': 'https://placehold.co/200x200/FBEC5D/000000?text=Corn',
#             'Rice': 'https://placehold.co/200x200/F5F5DC/000000?text=Rice',
#             'Sugarcane': 'https://placehold.co/200x200/90EE90/000000?text=Sugarcane',
#             'Cotton': 'https://placehold.co/200x200/FFFAFA/000000?text=Cotton',
#             'Default': 'https://placehold.co/200x200/CCCCCC/FFFFFF?text=Crop'
#         };

#         form.addEventListener('submit', async (e) => {
#             e.preventDefault();
#             const city = cityInput.value.trim();
#             if (!city) return;

#             // Reset UI
#             resultsContainer.style.display = 'none';
#             resultsContainer.classList.remove('fade-in');
#             errorMessage.style.display = 'none';
#             loader.style.display = 'flex';

#             try {
#                 const response = await fetch('/predict', {
#                     method: 'POST',
#                     headers: { 'Content-Type': 'application/json' },
#                     body: JSON.stringify({ city: city })
#                 });

#                 const data = await response.json();

#                 if (!response.ok) {
#                     throw new Error(data.error || 'Something went wrong');
#                 }
                
#                 updateUI(data);

#             } catch (error) {
#                 errorMessage.textContent = `Error: ${error.message}`;
#                 errorMessage.style.display = 'block';
#             } finally {
#                 loader.style.display = 'none';
#             }
#         });

#         function updateUI(data) {
#             // Forecast
#             document.getElementById('temp').textContent = `${data.forecast.temp_high}°C / ${data.forecast.temp_low}°C`;
#             document.getElementById('humidity').textContent = `${data.forecast.humidity}%`;
#             document.getElementById('wind').textContent = `${data.forecast.wind_speed} km/h`;

#             // Rainfall
#             document.getElementById('rain-prob').textContent = `${data.rainfall_prediction.rain_probability}%`;
#             document.getElementById('rain-mm').textContent = `${data.rainfall_prediction.rainfall_mm}`;
#             document.getElementById('rain-hours').textContent = `${data.rainfall_prediction.rain_duration_hours}`;

#             // Crop
#             document.getElementById('location').textContent = `${data.location.city}, ${data.location.state}`;
#             const cropName = data.crop_recommendation.crop;
#             document.getElementById('crop-name').textContent = cropName;
#             document.getElementById('crop-image').src = cropImages[cropName] || cropImages['Default'];
#             document.getElementById('crop-image').alt = cropName;

#             // Show results
#             resultsContainer.style.display = 'grid';
#             setTimeout(() => {
#                 resultsContainer.classList.add('fade-in');
#             }, 10);
#         }
#     </script>
# </body>
# </html>
# """

# # --- RUN THE APP ---
# if __name__ == '__main__':
#     # Train (or load) models on startup
#     train_weather_prediction_model()
#     train_crop_recommendation_model()
#     # Note: Use app.run(debug=True) for development to see errors and auto-reload.
#     # For production, use a proper WSGI server like Gunicorn or Waitress.
#     app.run(host='0.0.0.0', port=5001)

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
    
    dataset_path = "project-root/backend/app/weather_dataset_3000rows.csv"
    if not os.path.exists(dataset_path):
        print(f"FATAL ERROR: Weather dataset '{dataset_path}' not found.")
        print("Please ensure the dataset file is in the same directory as app.py.")
        exit()

    df = pd.read_csv(dataset_path)

    # Clean rain_probability (remove % and convert to float)
    df["rain_probability"] = df["rain_probability"].str.replace("%", "").astype(float)

    # Define target columns
    target_columns = ["rainfall_mm", "rain_duration_hours", "rain_probability", "temp_avg_c"]

    # Features & target split
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    weather_model_columns = X.columns # Save column order

    # Random Forest Multi-output model
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
        print("Please ensure the dataset file is in the same directory as app.py.")
        exit()
        
    df = pd.read_csv(dataset_path)
    
    # Separate features and target
    X = df.drop(columns=["recommended_crop"])
    y = df["recommended_crop"]

    # Encode target labels
    le_crop = LabelEncoder()
    y_encoded = le_crop.fit_transform(y)
    
    # One-hot encode the 'soil_type' feature
    X = pd.get_dummies(X, columns=['soil_type'], drop_first=False)

    # Store the column order from the training set
    X_train_cols = X.columns
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    crop_model = (model, X_train_cols) # Save model and columns
    crop_encoder = le_crop
    print("Crop recommendation model trained successfully from CSV.")


# --- DATA FETCHING & PROCESSING ---
def get_weather_and_rainfall_prediction(city_name):
    """
    Fetches weather data and predicts rainfall.
    """
    # 1. Get Coordinates
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
    geo_response = requests.get(geo_url)
    if geo_response.status_code != 200 or not geo_response.json():
        raise ValueError("Could not find coordinates for the city.")
    
    coords = geo_response.json()[0]
    lat, lon = coords['lat'], coords['lon']
    state = coords.get('state', 'N/A')

    # 2. Get 5-Day Forecast
    forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    forecast_response = requests.get(forecast_url)
    if forecast_response.status_code != 200:
        raise ValueError("Could not fetch weather forecast.")
    
    forecast_data = forecast_response.json()

    # 3. Process Forecast Data
    data = []
    for entry in forecast_data['list']:
        data.append({
            'temp_high_C': round(entry['main']['temp_max'] - 273.15, 1),
            'temp_low_C': round(entry['main']['temp_min'] - 273.15, 1),
            'humidity_%': entry['main']['humidity'],
            'wind_speed_mps': entry['wind']['speed']
        })
    
    df = pd.DataFrame(data)
    mean_values = df.mean()

    # 4. Prepare Data for Weather Prediction Model
    temp_high = round(mean_values["temp_high_C"], 1)
    temp_low = round(mean_values["temp_low_C"], 1)
    humidity = round(mean_values["humidity_%"], 1)
    wind_speed = round(mean_values["wind_speed_mps"] * 3.6, 1) # m/s to km/h

    month_map = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
    month_str = forecast_data['list'][0]['dt_txt'].split('-')[1]
    month = f'{month_map[month_str]}-25'

    weather_input_data = pd.DataFrame([{
        "state": state,
        "district": city_name,
        "month": month,
        "temp_high_c": temp_high,
        "temp_low_c": temp_low,
        "humidity_percent": humidity,
        "sunlight_hours": 8.0,
        "wind_speed_kmph": wind_speed,
        "wind_direction": "nW"
    }])
    
    weather_input_encoded = pd.get_dummies(weather_input_data)
    weather_input_aligned = weather_input_encoded.reindex(columns=weather_model_columns, fill_value=0)

    # 5. Predict Weather
    weather_prediction = weather_model.predict(weather_input_aligned)[0]
    
    results = {
        "location": {"city": city_name.title(), "state": state},
        "forecast": {
            "temp_high": temp_high,
            "temp_low": temp_low,
            "humidity": humidity,
            "wind_speed": wind_speed
        },
        "rainfall_prediction": {
            "rainfall_mm": max(0, round(weather_prediction[0], 1)),
            "rain_duration_hours": max(0, round(weather_prediction[1], 1)),
            "rain_probability": min(100, max(0, round(weather_prediction[2], 1))),
            "temp_avg": round(weather_prediction[3], 1)
        }
    }
    return results

def get_crop_recommendation(prediction_data):
    """
    Recommends a crop based on weather data and soil type.
    """
    # 1. Prepare Data for Crop Prediction Model
    crop_input_data = pd.DataFrame([{
        "soil_type": prediction_data["soil_type"],
        "rainfall_mm": prediction_data["rainfall_mm"],
        "temp_high_c": prediction_data["temp_high_c"],
        "temp_low_c": prediction_data["temp_low_c"],
        "temp_avg_c": prediction_data["temp_avg_c"],
        "humidity_percent": prediction_data["humidity_percent"],
        "sunlight_hours": 8.0,
        "wind_speed_kmph": prediction_data["wind_speed_kmph"]
    }])
    
    crop_input_encoded = pd.get_dummies(crop_input_data, columns=['soil_type'], drop_first=False)
    
    model, required_cols = crop_model
    crop_input_aligned = crop_input_encoded.reindex(columns=required_cols, fill_value=0)

    # 2. Predict Crop
    predicted_crop_encoded = model.predict(crop_input_aligned)[0]
    predicted_crop = crop_encoder.inverse_transform([predicted_crop_encoded])[0]

    return {"crop": predicted_crop}

# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    """Handles the weather prediction request."""
    city = request.json.get('city')
    if not city:
        return jsonify({"error": "City name is required."}), 400
    
    try:
        prediction_results = get_weather_and_rainfall_prediction(city)
        return jsonify(prediction_results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    """Handles the crop prediction request."""
    data = request.json
    required_keys = ["soil_type", "rainfall_mm", "temp_high_c", "temp_low_c", "temp_avg_c", "humidity_percent", "wind_speed_kmph"]
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing data for crop prediction."}), 400

    try:
        crop_result = get_crop_recommendation(data)
        return jsonify(crop_result)
    except Exception as e:
        print(f"An unexpected error occurred during crop prediction: {e}")
        return jsonify({"error": "An internal error occurred."}), 500


# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriCast - Weather & Crop Advisor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f4f8;
        }
        .card {
            background-color: white;
            border-radius: 1rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease, opacity 0.5s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        .icon { width: 50px; height: 50px; }
        .hidden-section { display: none; opacity: 0; }
        .visible-section { display: block; opacity: 1; }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
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
                <div class="card p-6 md:p-8 mb-8">
                    <form id="weather-form" class="flex flex-col sm:flex-row items-center gap-4">
                        <input type="text" id="city-input" name="city" placeholder="Enter your district or city name..." class="w-full px-4 py-3 text-lg border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-300" required>
                        <button type="submit" class="w-full sm:w-auto bg-blue-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-300 transition duration-300">
                            Get Forecast
                        </button>
                    </form>
                </div>

                <div id="loader" class="hidden justify-center items-center my-8">
                    <div class="loader"></div>
                    <p class="ml-4 text-gray-600">Fetching data and running AI models...</p>
                </div>

                <div id="error-message" class="hidden text-center text-red-600 bg-red-100 p-4 rounded-lg"></div>

                <div id="weather-results-container" class="hidden-section grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <!-- Weather Forecast Card -->
                    <div class="card p-6 col-span-1 md:col-span-1 flex flex-col">
                        <h2 class="text-xl font-semibold mb-4 text-gray-700">5-Day Average Forecast</h2>
                        <div class="flex items-center space-x-4 mb-4">
                           <svg class="icon text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" /></svg>
                           <div><p class="text-sm text-gray-500">Avg. High / Low</p><p id="temp" class="text-2xl font-bold"></p></div>
                        </div>
                        <div class="flex items-center space-x-4 mb-4">
                            <svg class="icon text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.362-3.797z" /></svg>
                            <div><p class="text-sm text-gray-500">Avg. Humidity</p><p id="humidity" class="text-2xl font-bold"></p></div>
                        </div>
                        <div class="flex items-center space-x-4">
                             <svg class="icon text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" /></svg>
                             <div><p class="text-sm text-gray-500">Avg. Wind Speed</p><p id="wind" class="text-2xl font-bold"></p></div>
                        </div>
                    </div>
                    <!-- Rainfall Prediction Card -->
                    <div class="card p-6 col-span-1 md:col-span-2 flex flex-col">
                        <h2 class="text-xl font-semibold mb-4 text-gray-700">Rainfall Prediction</h2>
                        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
                            <div><p class="text-sm text-gray-500">Probability</p><p id="rain-prob" class="text-3xl font-bold text-blue-600"></p></div>
                            <div><p class="text-sm text-gray-500">Amount (mm)</p><p id="rain-mm" class="text-3xl font-bold text-blue-600"></p></div>
                            <div><p class="text-sm text-gray-500">Duration (hrs)</p><p id="rain-hours" class="text-3xl font-bold text-blue-600"></p></div>
                        </div>
                    </div>
                </div>

                <!-- Crop Prediction Section -->
                <div id="crop-prediction-section" class="hidden-section card p-6 md:p-8">
                    <h2 class="text-2xl font-semibold mb-4 text-gray-800">Get Crop Recommendation</h2>
                    <form id="crop-form" class="flex flex-col sm:flex-row items-center gap-4">
                        <select id="soil-type-select" class="w-full px-4 py-3 text-lg border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition duration-300">
                            <option value="Loamy">Loamy Soil</option>
                            <option value="Sandy">Sandy Soil</option>
                            <option value="Clay">Clay Soil</option>
                            <option value="Black">Black Soil</option>
                            <option value="Red">Red Soil</option>
                        </select>
                        <button type="submit" class="w-full sm:w-auto bg-green-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-4 focus:ring-green-300 transition duration-300">
                            Predict Crop
                        </button>
                    </form>
                </div>
                
                <div id="crop-loader" class="hidden justify-center items-center my-8">
                    <div class="loader"></div>
                    <p class="ml-4 text-gray-600">Recommending the best crop...</p>
                </div>

                <!-- Crop Recommendation Card -->
                <div id="crop-result-card" class="hidden-section card p-6 mt-6 col-span-1 md:col-span-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white">
                     <div class="flex flex-col md:flex-row items-center text-center md:text-left">
                        <div class="mb-4 md:mb-0 md:mr-6">
                            <p class="text-lg">Based on the forecast for <strong id="location" class="font-bold"></strong> and your selected soil, the recommended crop is:</p>
                            <p id="crop-name" class="text-5xl font-extrabold tracking-tight mt-2"></p>
                        </div>
                        <div class="md:ml-auto">
                            <img id="crop-image" src="" alt="Recommended Crop" class="w-32 h-32 object-cover rounded-full shadow-lg border-4 border-white/50">
                        </div>
                    </div>
                </div>

            </main>
        </div>
    </div>

    <script>
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

        const cropImages = {
            'Wheat': 'https://placehold.co/200x200/F4A460/FFFFFF?text=Wheat',
            'Corn': 'https://placehold.co/200x200/FBEC5D/000000?text=Corn',
            'Rice': 'https://placehold.co/200x200/F5F5DC/000000?text=Rice',
            'Sugarcane': 'https://placehold.co/200x200/90EE90/000000?text=Sugarcane',
            'Cotton': 'https://placehold.co/200x200/FFFAFA/000000?text=Cotton',
            'Bajra': 'https://placehold.co/200x200/BDB76B/FFFFFF?text=Bajra',
            'Groundnut': 'https://placehold.co/200x200/D2B48C/FFFFFF?text=Groundnut',
            'Jowar': 'https://placehold.co/200x200/DAA520/FFFFFF?text=Jowar',
            'Maize': 'https://placehold.co/200x200/FFD700/000000?text=Maize',
            'Soyabean': 'https://placehold.co/200x200/ADFF2F/000000?text=Soyabean',
            'Default': 'https://placehold.co/200x200/CCCCCC/FFFFFF?text=Crop'
        };

        weatherForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const city = cityInput.value.trim();
            if (!city) return;

            // Reset UI
            hideAllResults();
            errorMessage.style.display = 'none';
            loader.style.display = 'flex';

            try {
                const response = await fetch('/predict_weather', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ city })
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Something went wrong');
                
                currentWeatherData = data; // Store weather data
                updateWeatherUI(data);

            } catch (error) {
                errorMessage.textContent = `Error: ${error.message}`;
                errorMessage.style.display = 'block';
            } finally {
                loader.style.display = 'none';
            }
        });

        cropForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!currentWeatherData) return;
            
            cropResultCard.classList.remove('visible-section');
            cropResultCard.classList.add('hidden-section');
            errorMessage.style.display = 'none';
            cropLoader.style.display = 'flex';

            const payload = {
                soil_type: soilTypeSelect.value,
                rainfall_mm: currentWeatherData.rainfall_prediction.rainfall_mm,
                temp_high_c: currentWeatherData.forecast.temp_high,
                temp_low_c: currentWeatherData.forecast.temp_low,
                temp_avg_c: currentWeatherData.rainfall_prediction.temp_avg,
                humidity_percent: currentWeatherData.forecast.humidity,
                wind_speed_kmph: currentWeatherData.forecast.wind_speed,
            };

            try {
                const response = await fetch('/predict_crop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Something went wrong');

                updateCropUI(data);

            } catch (error) {
                 errorMessage.textContent = `Error: ${error.message}`;
                 errorMessage.style.display = 'block';
            } finally {
                cropLoader.style.display = 'none';
            }
        });

        function hideAllResults() {
            weatherResultsContainer.classList.remove('visible-section');
            weatherResultsContainer.classList.add('hidden-section');
            cropPredictionSection.classList.remove('visible-section');
            cropPredictionSection.classList.add('hidden-section');
            cropResultCard.classList.remove('visible-section');
            cropResultCard.classList.add('hidden-section');
        }

        function updateWeatherUI(data) {
            // Forecast
            document.getElementById('temp').textContent = `${data.forecast.temp_high}°C / ${data.forecast.temp_low}°C`;
            document.getElementById('humidity').textContent = `${data.forecast.humidity}%`;
            document.getElementById('wind').textContent = `${data.forecast.wind_speed} km/h`;

            // Rainfall
            document.getElementById('rain-prob').textContent = `${data.rainfall_prediction.rain_probability}%`;
            document.getElementById('rain-mm').textContent = `${data.rainfall_prediction.rainfall_mm}`;
            document.getElementById('rain-hours').textContent = `${data.rainfall_prediction.rain_duration_hours}`;

            // Show weather results and crop input section
            weatherResultsContainer.classList.remove('hidden-section');
            cropPredictionSection.classList.remove('hidden-section');
            setTimeout(() => {
                weatherResultsContainer.classList.add('visible-section');
                cropPredictionSection.classList.add('visible-section');
            }, 10);
        }

        function updateCropUI(data) {
            document.getElementById('location').textContent = `${currentWeatherData.location.city}, ${currentWeatherData.location.state}`;
            const cropName = data.crop;
            document.getElementById('crop-name').textContent = cropName;
            document.getElementById('crop-image').src = cropImages[cropName] || cropImages['Default'];
            document.getElementById('crop-image').alt = cropName;
            
            cropResultCard.classList.remove('hidden-section');
            setTimeout(() => {
                 cropResultCard.classList.add('visible-section');
            }, 10);
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
    # Note: Use app.run(debug=True) for development to see errors and auto-reload.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(host='0.0.0.0', port=5001)


AgriCast - AI-Powered Weather & Crop Advisor

AgriCast is an intelligent web application designed to provide farmers and agricultural enthusiasts with real-time weather forecasts and AI-driven crop recommendations. By simply entering a district or city, users receive a 5-day average weather forecast and a prediction for rainfall. Based on this forecast, they can then select a soil type to get a scientifically-backed recommendation for the most suitable crop to plant.
The application leverages the OpenWeatherMap API for live weather data and uses two distinct machine learning models trained on comprehensive agricultural datasets for its predictions.

Features
* Dynamic Weather Forecasting: Fetches and displays a 5-day average forecast (temperature, humidity, wind speed) for any city/district.
* AI Rainfall Prediction: Utilizes a RandomForestRegressor model to predict rainfall probability, amount (mm), and duration (hours).
* AI Crop Recommendation: Employs a RandomForestClassifier model to recommend the optimal crop based on the weather forecast and user-selected soil type.
* Interactive UI: A clean, responsive, two-step user interface that first provides weather data and then prompts for the information needed for a crop prediction.
* Real-world Data: The models are trained on weather_dataset_3000rows.csv and diverse_crop_suitability_balanced_with_soil.csv to ensure relevant and accurate predictions.
Tech Stack
* Backend: Flask, Pandas, Scikit-learn, NumPy, Requests
* Frontend: HTML, Tailwind CSS, vanilla JavaScript
* Data Source: OpenWeatherMap API
  
Project Structure
Your project needs to be organized in a specific way for the application to find the dataset files correctly. Please create the following folder structure:

project-root/
├── backend/
│   └── app/
│       ├── app.py
│       ├── diverse_crop_suitability_balanced_with_soil.csv
│       └── weather_dataset_3000rows.csv
└── requirements.txt

You should run the application from the project-root directory.

Setup and Installation

Follow these steps to get the AgriCast application running on your local machine.
1. Prerequisites
* Python 3.7 or newer
* pip (Python package installer)
2. Clone the Repository
Clone your project repository to your local machine. If you haven't created one yet, see the "How to Push to GitHub" section below.
git clone <your-repository-url>
cd project-root

3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

4. Install Dependencies
Install all the required Python libraries using the requirements.txt file.
pip install -r requirements.txt

5. Get an OpenWeatherMap API Key
The application requires an API key from OpenWeatherMap to fetch weather data.
* Sign up for a free account.
* Find your API key on your account page.
* Open the backend/app/app.py file and replace the placeholder value of the API_KEY variable with your actual key.
# In backend/app/app.py
API_KEY = "your_actual_api_key_here" 

6. Run the Application
Navigate to the project's root directory and run the Flask application.
# Make sure you are in the 'project-root' directory
python backend/app/app.py

You should see output indicating that the models have been trained and the server is running, typically on http://127.0.0.1:5001. Open this URL in your web browser to use the application.
How to Push Your Project to GitHub
If you haven't already, follow these steps to upload your project to a new GitHub repository.

Step 1: Initialize Git
If you haven't done so, initialize a Git repository in your project-root folder.
git init

Step 2: Add and Commit Your Files
Add all your project files to the staging area and make your first commit.
# Add all files in the current directory
git add .

# Commit the files with a message
git commit -m "Initial commit: AgriCast application setup"

Step 3: Create a New Repository on GitHub
Go to GitHub and create a new, empty repository. Do not initialize it with a README or .gitignore file.
Step 4: Link Your Local Repository to GitHub
Copy the URL of your new GitHub repository. Then, in your terminal, link your local repository to the remote one on GitHub.
git remote add origin <your-new-repository-url.git>

Step 5: Push Your Code
Push your committed files to GitHub. The -u flag sets the upstream branch for future pushes.
git push -u origin main

Your code is now live on GitHub!

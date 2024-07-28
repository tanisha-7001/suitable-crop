import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

data = pd.read_csv('Crop_Recommendation.csv')

data = data.dropna()

le = LabelEncoder()
data['Crop'] = le.fit_transform(data['Crop'])

features = data.drop('Crop', axis=1)
target = data['Crop']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt.gcf())
    plt.clf()

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

yield_data = {
    'Rice': 4.0,           # tons per hectare
    'Maize': 5.5,          # tons per hectare
    'ChickPea': 1.2,       # tons per hectare
    'KidneyBeans': 1.8,    # tons per hectare
    'PigeonPeas': 1.0,     # tons per hectare
    'MothBeans': 0.6,      # tons per hectare
    'MungBean': 0.8,       # tons per hectare
    'Blackgram': 0.7,      # tons per hectare
    'Lentil': 1.0,         # tons per hectare
    'Pomegranate': 10.0,   # tons per hectare
    'Banana': 40.0,        # tons per hectare
    'Mango': 10.0,         # tons per hectare
    'Grapes': 25.0,        # tons per hectare
    'Watermelon': 30.0,    # tons per hectare
    'Muskmelon': 15.0,     # tons per hectare
    'Apple': 20.0,         # tons per hectare
    'Orange': 20.0,        # tons per hectare
    'Papaya': 40.0,        # tons per hectare
    'Coconut': 2.0,        # tons per hectare (converted from nuts)
    'Cotton': 1.5,         # tons per hectare (converted from bales)
    'Jute': 2.5,           # tons per hectare
    'Coffee': 2.0          # tons per hectare
}


st.title('Crop Recommendation System')

st.write('Enter the following parameters to get a crop recommendation:')

nitrogen = st.number_input('Nitrogen', min_value=0, max_value=100)
phosphorus = st.number_input('Phosphorus', min_value=0, max_value=100)
potassium = st.number_input('Potassium', min_value=0, max_value=100)
temperature = st.number_input('Temperature', min_value=0.0, max_value=50.0)
humidity = st.number_input('Humidity', min_value=0.0, max_value=100.0)
ph_value = st.number_input('pH Value', min_value=0.0, max_value=14.0)
rainfall = st.number_input('Rainfall', min_value=0.0, max_value=300.0)
area = st.number_input('Area of land (hectares)', min_value=0.1, max_value=1000.0)

if st.button('Predict Crop'):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
    prediction = rf_clf.predict(input_data)
    crop_name = le.inverse_transform(prediction)
    st.write(f'The recommended crop is: {crop_name[0]}')

    estimated_yield_per_hectare = yield_data.get(crop_name[0], 0)
    estimated_yield = estimated_yield_per_hectare * area
    st.write(f'Estimated yield for {area} hectares of {crop_name[0]}: {estimated_yield} tons')


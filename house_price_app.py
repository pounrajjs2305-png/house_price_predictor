import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
explainer = shap.TreeExplainer(model)

feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longtitude"
]

importances = model.feature_importances_

feature_importance = pd.Series(importances, index=feature_names)
feature_importance = feature_importance.sort_values(ascending=False)

st.title("🏠 House Price Predictor")

st.subheader("Feature Importance")

fig, ax = plt.subplots()
feature_importance.plot(kind="bar", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.write("Enter details to predict house price")

# Inputs
MedInc = st.number_input("Median Income", 0.0, 20.0, 5.0)
HouseAge = st.number_input("House Age", 1, 100, 20)
AveRooms = st.number_input("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.number_input("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.number_input("Population", 100, 50000, 1000)
AveOccup = st.number_input("Average Occupancy", 1.0, 10.0, 3.0)
Latitude = st.number_input("Latitude", 30.0, 45.0, 35.0)
Longitude = st.number_input("Longitude", -125.0, -110.0, -120.0)

#Visualization
st.subheader("Price vs Income Analysis")

#Generate range of income values
income_range = np.linspace(1, 10, 50)

predictions = []

#Generate predictions
for inc in income_range:
    sample = np.array([[
        inc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])

    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    predictions.append(pred)

#plot
fig, ax = plt.subplots()
ax.plot(income_range, predictions)
ax.set_xlabel("Median Income")
ax.set_ylabel("Predicted Price")
ax.set_title("Price vs Income")
st.pyplot(fig)

st.subheader("Input Summary")

st.write({
    "Median Income": MedInc,
    "House Age": HouseAge,
    "Rooms": AveRooms,
    "Bedrooms": AveBedrms,
    "Population": Population,
    "Occupancy": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
}) 

# Predict button
if st.button("Predict Price"):

    sample = np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    st.success(f"Estimated House Price: ${prediction * 100000:,.2f}")

if st.button("Explain Prediction"):

    sample = np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])

    sample_scaled = scaler.transform(sample)

    shap_values = explainer(sample_scaled)

    st.subheader("SHAP Explanation") 

    #Plot
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    
# house_price_predictor
ML app to predict house prices using Random Forest and SHAP
This project is a Machine Learning web app built using streamlit
Features
Predict house prices using Random Forest
Feature importance visualization
Price vs Income analysis graph
SHAP-based explainability(Explainable AI) 
Model
Random Forest Regressor
Trained on California Housing Dataset
Technologies used
Python 
scikit-learn
SHAP
Matplotlib
Hoe to Run
pip install -r requirements.txt
sreamlit run house_price_app.py
Output
Interactive UI where users can:
Enter house details
predict price
Visualize model behaviour
Undersatnd predictions
Author 
PounrajJS
The trained model file is not included due to size limits.
To run this project 
Train the model using the provided notebook/code
Save using joblib:
joblib.dump(model, "house_price_model.pkl")
place it in the project folder

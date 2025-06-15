# Car-Price-Prediction-ML
## 📌 Overview
This project uses a machine learning model to predict the selling price of used cars based on features like present price, kilometers driven, fuel type, transmission, and more. The dataset is preprocessed with feature engineering and label encoding. A linear regression model is trained and evaluated using R² score and visualization


## ✨ Features

- **Data Analysis**: Comprehensive exploratory data analysis (EDA) with visualizations
- **Feature Engineering**: Creation of new relevant features (Car Age, Price Drop, Km/Year)
- **Multiple Models**: Comparison of Linear Regression and Random Forest models
- **Hyperparameter Tuning**: Optimized model performance using GridSearchCV
- **Evaluation Metrics**: R² Score, MAE, RMSE for model evaluation
- **Visualizations**: Actual vs Predicted prices, residual analysis, feature importance
- **Model Deployment**: Save trained model for future use

## 📊 Dataset

The dataset contains information about used cars with the following features:

| Feature | Description |
|---------|-------------|
| Car_Name | Name of the car |
| Year | Year of purchase |
| Selling_Price | Current selling price (target variable) |
| Present_Price | Original showroom price |
| Driven_kms | Kilometers driven |
| Fuel_Type | Fuel type (Petrol/Diesel/CNG) |
| Seller_Type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/laiba-Ashfaq/Car-Price-Prediction-ML.git
   cd Car-Price-Prediction
Install the required packages:

bash
pip install -r requirements.txt
Or install manually:

bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
🚀 Usage
Run the Jupyter notebook or Python script:

bash
python car_price_prediction.py
The script will:

Load and preprocess the data

Train and evaluate models

Generate visualizations

Save the best model as best_car_price_model.pkl

To make predictions with the saved model:

python
import joblib
import pandas as pd

# Load the model
model = joblib.load('best_car_price_model.pkl')

# Prepare input data (example)
input_data = pd.DataFrame({
    'Present_Price': [9.85],
    'Driven_kms': [6900],
    'Fuel_Type': [0],  # 0:Petrol, 1:Diesel, 2:CNG
    'Seller_Type': [0],  # 0:Dealer, 1:Individual
    'Transmission': [0],  # 0:Manual, 1:Automatic
    'Owner': [0],
    'Car_Age': [8],
    'Price_Drop': [2.6],
    'Km_Per_Year': [862.5]
})

# Make prediction
predicted_price = model.predict(input_data)
print(f"Predicted Selling Price: {predicted_price[0]:.2f} lakhs")
📈 Results
Model Performance Comparison
Model	R² Score	MAE	RMSE
Linear Regression	0.85	1.23	2.15
Random Forest	0.92	0.78	1.45
Visualizations
# Actual vs Predicted Prices and Feature Importance:
![Screenshot 2025-06-16 024717](https://github.com/user-attachments/assets/f101c31e-e2fa-4f23-b47c-def62735af04)
# Residual Analysis
![Screenshot 2025-06-16 024746](https://github.com/user-attachments/assets/2195bfb5-d5fc-472e-956f-31a263d63ed1)
![Screenshot 2025-06-16 024759](https://github.com/user-attachments/assets/f82d1353-b604-4b93-844f-06ab0446306c)



# 📂 Project Structure
Car-Price-Prediction/
├── data/
│   └── car data.csv          # Dataset                 
├── code.py                   # Main Python script
└── README.md                 # Project documentation

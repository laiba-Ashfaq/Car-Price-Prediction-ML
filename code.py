import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("car data.csv")

# Display basic info
print("=== Dataset Head ===")
print(df.head())
print("\n=== Dataset Info ===")
print(df.info())
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Data Cleaning
# Fix typo in 'wagon r' year (assuming it should be 2011 like others)
df.loc[df['Car_Name'] == 'wagon r', 'Year'] = 2011

# Rename column for consistency
df.rename(columns={'Selling_type': 'Seller_Type'}, inplace=True)

# Feature Engineering
# Create new features
current_year = 2025
df['Car_Age'] = current_year - df['Year']
df['Price_Drop'] = df['Present_Price'] - df['Selling_Price']
df['Km_Per_Year'] = df['Driven_kms'] / np.where(df['Car_Age'] == 0, 1, df['Car_Age'])  # Avoid division by zero

# Encode categorical variables
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
df['Seller_Type'] = df['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})

# Drop unneeded columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# EDA Visualization
plt.figure(figsize=(15, 10))

# Correlation heatmap
plt.subplot(2, 2, 1)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')

# Selling price distribution
plt.subplot(2, 2, 2)
sns.histplot(df['Selling_Price'], kde=True)
plt.title('Selling Price Distribution')

# Price vs Age
plt.subplot(2, 2, 3)
sns.scatterplot(x='Car_Age', y='Selling_Price', data=df, hue='Fuel_Type')
plt.title('Price vs Car Age')

# Price vs Kilometers Driven
plt.subplot(2, 2, 4)
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=df, hue='Transmission')
plt.title('Price vs Kilometers Driven')

plt.tight_layout()
plt.show()

# Define features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Pipeline
# Linear Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning for Random Forest
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

# Train and evaluate models
models = {
    'Linear Regression': lr_pipeline,
    'Random Forest': grid_search
}

results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    
    if name == 'Random Forest':
        print("Best parameters:", model.best_params_)
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results[name] = {
        'R2 Score': r2,
        'MAE': mae,
        'RMSE': rmse,
        'Model': model
    }
    
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

# Compare models
print("\n=== Model Comparison ===")
results_df = pd.DataFrame(results).T
print(results_df[['R2 Score', 'MAE', 'RMSE']])

# Feature Importance
best_rf = results['Random Forest']['Model'].named_steps['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# Best model predictions visualization
best_model = results['Random Forest']['Model']
y_pred = best_model.predict(X_test)

plt.figure(figsize=(12, 6))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Analysis')

plt.tight_layout()
plt.show()

# Save the best model
import joblib
joblib.dump(best_model, 'best_car_price_model.pkl')
print("\nBest model saved as 'best_car_price_model.pkl'")
# train.py

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("crop_yield.csv")

# Inspect the data (print first few rows)
print(df.head())

# Handle missing values: impute numeric columns with mean, categorical with most frequent value
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Impute numeric columns (mean strategy)
numeric_imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute categorical columns (most frequent strategy)
categorical_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Encode categorical columns (State, Crop, etc.)
label_cols = ['State', 'Crop']  # Add other categorical columns as needed
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert 'Season' into numeric form
df['Season'] = df['Season'].map({'Whole Year': 1, 'Rabi': 2, 'Kharif': 3, 'Zaid': 4}).fillna(0)

# Features and target
X = df.drop(['Yield'], axis=1)
y = df['Yield']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'XGBoost': {
        'n_estimators': [100, 200, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'LightGBM': {
        'n_estimators': [100, 200, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
    }
}

best_models = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid = GridSearchCV(estimator=model, param_grid=param_grid[model_name], cv=3, scoring='r2', verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[model_name] = grid.best_estimator_

    # Evaluate model
    y_pred = grid.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters for {model_name}: {grid.best_params_}")
    print(f"Mean Squared Error for {model_name}: {mse:.2f}")
    print(f"R² Score for {model_name}: {r2:.4f}")
    
    # Save the best models
    joblib.dump(grid.best_estimator_, f"{model_name}_crop_yield_model.pkl")
    
# Select the best performing model (based on R² score)
best_model_name = max(best_models, key=lambda x: r2_score(y_test, best_models[x].predict(X_test)))
best_model = best_models[best_model_name]
print(f"Best performing model is: {best_model_name} with R² score of {r2_score(y_test, best_model.predict(X_test)):.4f}")

# Save the best performing model
joblib.dump(best_model, "best_crop_yield_model.pkl")

# Save preprocessing artifacts
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model and preprocessing artifacts saved.")

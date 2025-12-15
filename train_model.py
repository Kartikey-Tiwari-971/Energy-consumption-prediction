import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# 1. Generate Synthetic Data
np.random.seed(42)
n_samples = 2000

# Define appliance categories with typical ranges for count and wattage
data = {
    # Lights: Often many, low wattage (e.g., LED 10W - 100W)
    'n_lights': np.random.randint(2, 20, n_samples),
    'w_lights': np.random.uniform(5, 50, n_samples),

    # Fans: Few, medium wattage (50W - 100W)
    'n_fans': np.random.randint(0, 6, n_samples),
    'w_fans': np.random.uniform(40, 100, n_samples),

    # AC: Rare, high wattage (1000W - 2500W)
    'n_ac': np.random.randint(0, 4, n_samples),
    'w_ac': np.random.uniform(1000, 2500, n_samples),

    # TV: One or two, medium wattage (50W - 200W)
    'n_tv': np.random.randint(0, 3, n_samples),
    'w_tv': np.random.uniform(50, 200, n_samples),

    # Fridge: Usually one, continuously running but cycles (100W - 500W rated)
    'n_fridge': np.random.randint(1, 3, n_samples),
    'w_fridge': np.random.uniform(100, 400, n_samples),
}

df = pd.DataFrame(data)

def calculate_daily_consumption(row):
    # Simulated usage hours (randomized per household/sample to create variance)
    # These distributions represent "typical" behavior the model will learn

    # Lights: Evening/Night use (approx 4-8 hours)
    h_lights = np.random.uniform(4, 8)

    # Fans: Depends on climate, assume 8-16 hours
    h_fans = np.random.uniform(8, 16)

    # AC: Intermittent, assume 4-10 hours if present
    h_ac = np.random.uniform(4, 10)

    # TV: Leisure, 2-6 hours
    h_tv = np.random.uniform(2, 6)

    # Fridge: Runs 24h but compressor cycles. Rated power is peak.
    # Effective full-load hours approx 8-12 hours/day
    h_fridge = np.random.uniform(8, 12)

    # Energy (kWh) = (Count * Wattage * Hours) / 1000
    kwh = 0
    kwh += (row['n_lights'] * row['w_lights'] * h_lights) / 1000
    kwh += (row['n_fans'] * row['w_fans'] * h_fans) / 1000
    kwh += (row['n_ac'] * row['w_ac'] * h_ac) / 1000
    kwh += (row['n_tv'] * row['w_tv'] * h_tv) / 1000
    kwh += (row['n_fridge'] * row['w_fridge'] * h_fridge) / 1000

    return kwh

# Calculate target variable with noise
# The noise represents user behavioral randomness not captured by simple "hours" averages
df['daily_consumption'] = df.apply(calculate_daily_consumption, axis=1) * np.random.uniform(0.9, 1.1, n_samples)

print("First 5 rows of synthetic data:")
print(df.head())

# 2. Train Model
feature_cols = ['n_lights', 'w_lights', 'n_fans', 'w_fans',
                'n_ac', 'w_ac', 'n_tv', 'w_tv',
                'n_fridge', 'w_fridge']
X = df[feature_cols]
y = df['daily_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f} kWh")
print(f"Root Mean Squared Error: {rmse:.2f} kWh")

# 4. Save Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to model.pkl")

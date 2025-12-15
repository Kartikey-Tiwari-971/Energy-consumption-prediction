import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# 1. Generate Synthetic Data
np.random.seed(42)
n_samples = 1000

data = {
    'temperature': np.random.uniform(10, 35, n_samples),
    'humidity': np.random.uniform(30, 90, n_samples),
    'square_footage': np.random.uniform(500, 5000, n_samples),
    'occupancy': np.random.randint(1, 10, n_samples),
    'hour_of_day': np.random.randint(0, 24, n_samples)
}

df = pd.DataFrame(data)

# Define a function to generate energy consumption based on features
def calculate_energy(row):
    # Base consumption
    energy = 5.0

    # Temperature effect (more energy used if too hot or too cold)
    # Assuming ideal temp is 22C. Deviation increases energy use (AC/Heating)
    energy += abs(row['temperature'] - 22) * 0.5

    # Square footage effect
    energy += row['square_footage'] * 0.002

    # Occupancy effect
    energy += row['occupancy'] * 0.5

    # Time of day effect (e.g., peak hours 17-21)
    if 17 <= row['hour_of_day'] <= 21:
        energy += 2.0
    elif 9 <= row['hour_of_day'] <= 17:
        energy += 1.0

    return energy

# Calculate target variable with some noise
df['energy_consumption'] = df.apply(calculate_energy, axis=1) + np.random.normal(0, 0.5, n_samples)

print("First 5 rows of synthetic data:")
print(df.head())

# 2. Train Model
X = df[['temperature', 'humidity', 'square_footage', 'occupancy', 'hour_of_day']]
y = df['energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# 4. Save Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to model.pkl")

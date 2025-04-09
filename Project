import pandas as pd
import sklearn as skl
file1=pd.read_csv("/content/drive/MyDrive/CO2 Emissions_Canada.csv")
# Neural Network
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Select features and target
X = file1[['Engine Size(L)', 'Fuel Consumption Comb (L/100 km)']]
y = file1['CO2 Emissions(g/km)']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the neural network model
# Hidden layers: (100, 50) means two hidden layers with 100 and 50 neurons respectively
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Plotting Actual vs. Predicted CO2 Emissions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs. Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')

# Add titles and labels
plt.title("Neural Network: Actual vs. Predicted CO₂ Emissions")
plt.xlabel("Actual CO₂ Emissions (g/km)")
plt.ylabel("Predicted CO₂ Emissions (g/km)")
plt.legend()
plt.grid(True)
plt.show()

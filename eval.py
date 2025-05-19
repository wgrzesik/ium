import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = load_model("data/airbnb_price_model.h5")

test_data = pd.read_csv("data/AB_NYC_2019_test.csv")

X_test = test_data.drop('price', axis=1)
y_test = test_data['price']

predictions = model.predict(X_test)
predictions = predictions.flatten()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

with open("regression_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

results = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
results.to_csv('data/airbnb_price_predictions.csv', index=False)
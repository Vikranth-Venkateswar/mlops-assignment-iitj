import joblib
from utils import load_data

_, X_test, _, _ = load_data()
model = joblib.load("model.joblib")

preds = model.predict(X_test[:5])
print("Sample Predictions:", preds)

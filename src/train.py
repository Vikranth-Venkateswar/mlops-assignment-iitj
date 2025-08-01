import joblib
from utils import load_data, create_model, evaluate_model

X_train, X_test, y_train, y_test = load_data()
model = create_model()
model.fit(X_train, y_train)

r2, mse = evaluate_model(model, X_test, y_test)
print(f"R2 Score: {r2}")
print(f"MSE: {mse}")

joblib.dump(model, "model.joblib")

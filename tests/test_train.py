import joblib
from sklearn.linear_model import LinearRegression
from src.utils import load_data, create_model

def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

def test_model_creation():
    model = create_model()
    assert isinstance(model, LinearRegression)

def test_model_training():
    X_train, X_test, y_train, y_test = load_data()
    model = create_model()
    model.fit(X_train, y_train)
    assert hasattr(model, "coef_")

def test_r2_score_threshold():
    from src.utils import evaluate_model
    X_train, X_test, y_train, y_test = load_data()
    model = create_model()
    model.fit(X_train, y_train)
    r2, _ = evaluate_model(model, X_test, y_test)
    assert r2 > 0.4  # reasonable threshold

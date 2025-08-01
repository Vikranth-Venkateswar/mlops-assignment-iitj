import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data():
    data = fetch_california_housing()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)

def create_model():
    return LinearRegression()

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    return r2, mse
def fetch_data_split():
    """Retrieve and split the California Housing dataset."""
    dataset = fetch_california_housing()
    features, targets = dataset.data, dataset.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    return X_tr, X_te, y_tr, y_te

def compute_scores(actual, predicted):
    """Compute R2 and MSE metrics."""
    score_r2 = r2_score(actual, predicted)
    score_mse = mean_squared_error(actual, predicted)
    return score_r2, score_mse

def compress_to_uint8(arr):
    """Quantize float array to uint8 using per-element min-max scaling."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        min_v = arr
        max_v = arr
        if arr.size == 1:
            min_v = max_v = arr
        else:
            min_v = arr.copy()
            max_v = arr.copy()
        min_v = arr
        max_v = arr
    else:
        min_v = arr.min(axis=0)
        max_v = arr.max(axis=0)
    min_v = np.asarray(min_v)
    max_v = np.asarray(max_v)
    quant = np.zeros_like(arr, dtype=np.uint8)
    for idx in range(arr.size):
        val = arr.flat[idx]
        mn = min_v.flat[idx]
        mx = max_v.flat[idx]
        if mx == mn:
            quant.flat[idx] = 0
        else:
            scale = 255.0 / (mx - mn)
            quant.flat[idx] = np.round((val - mn) * scale).astype(np.uint8)
    return quant, min_v.astype(float), max_v.astype(float)


def decompress_from_uint8(quant, min_v, max_v):
    """Dequantize uint8 array back to float using per-element min and max."""
    quant = np.asarray(quant, dtype=np.uint8)
    min_v = np.asarray(min_v)
    max_v = np.asarray(max_v)
    dequant = np.zeros_like(quant, dtype=np.float32)
    for idx in range(quant.size):
        mn = min_v.flat[idx]
        mx = max_v.flat[idx]
        if mx == mn:
            dequant.flat[idx] = mn
        else:
            scale = (mx - mn) / 255.0
            dequant.flat[idx] = quant.flat[idx] * scale + mn
    return dequant

# MLOps Linear Regression Pipeline

This demonstrates a full MLOps pipeline using a simple Linear Regression model on the California Housing dataset. The goal was to set up not just model training, but also quantization, Docker deployment, and CI/CD using GitHub Actions — all in a clean and reproducible way.

---

## What’s Included

* Loading and splitting the dataset using `scikit-learn`
* Training a Linear Regression model
* Saving and reloading the model using `joblib`
* Manually quantizing the model weights to `uint8`
* Dequantizing and evaluating the quantized model
* Writing test cases to verify model correctness
* Building a Docker image and running predictions inside a container
* Automating everything using GitHub Actions

---

## Folder Structure

```
.
├── src/                  # All source scripts
│   ├── train.py          # Model training
│   ├── predict.py        # Inference script
│   ├── quantize.py       # Quantization logic
│   └── utils.py          # Common functions
├── tests/                # Unit tests
│   └── test_train.py
├── Dockerfile            # Docker setup
├── requirements.txt      # Python dependencies
├── .github/workflows/    # CI/CD workflow file
│   └── ci.yml
└── README.md             # This file
```

---

## How to Run the Project

Make sure you have Python 3.10+ installed.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Quantize and test

```bash
python src/quantize.py
```

### 4. Run predictions

```bash
python src/predict.py
```

### 5. Run tests

```bash
pytest tests/
```

---

## Docker Instructions

To test everything inside a Docker container:

```bash
docker build -t mlops-lr .
docker run mlops-lr
```

This will run the prediction script and print out a few results.

---

## CI/CD

The `.github/workflows/ci.yml` file sets up a 3-step GitHub Actions pipeline:

* Run unit tests
* Train and quantize the model
* Build and test the Docker container

All of this runs automatically on every push to the `main` branch.

---

## Model Performance

| Metric               | Value  |
| -------------------- | ------ |
| R2 Score             | 0.5758 |
| MSE                  | 0.5559 |
| Max Pred Error (QZ)  | 9.8753 |
| Mean Pred Error (QZ) | 0.5332 |


---





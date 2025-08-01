FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

# Train the model and then run predictions
CMD python src/train.py && python src/predict.py

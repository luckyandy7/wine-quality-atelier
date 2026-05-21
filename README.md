# Wine Quality Atelier

A complete local wine-quality prediction project built from the UCI red wine quality dataset.

The app trains a scikit-learn model from `data/winequality-red.csv`, serves a JSON API, and provides a modern wine-shop interface for exploring predicted quality.

## Run

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:8765
```

If `models/wine_quality_bundle.joblib` does not exist, the app trains it automatically on startup.

## API

```text
GET  /api/health
GET  /api/metadata
POST /api/predict
POST /api/retrain
```

`POST /api/predict` expects all eleven wine chemistry features:

```json
{
  "fixed acidity": 7.4,
  "volatile acidity": 0.7,
  "citric acid": 0.0,
  "residual sugar": 1.9,
  "chlorides": 0.076,
  "free sulfur dioxide": 11,
  "total sulfur dioxide": 34,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

## Test

```powershell
python -m unittest discover -s tests
```

## Source Data

The dataset is copied from the local file:

```text
C:\Users\lucky\Downloads\머신러닝 프로젝트\winequality-red.csv
```

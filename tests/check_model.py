import joblib
import pandas as pd
import sys

pipeline = joblib.load("models/modelo_top10.pkl")
X = pd.DataFrame([{
    "grid": 3, "round": 5, "year": 2023,
    "TrackTemp": 35.0, "AirTemp": 28.0, "Humidity": 45.0,
    "Rainfall": 0.0, "WindSpeed": 12.0,
    "constructorId": 9, "circuitId": 14,
}])
pred = int(pipeline.predict(X)[0])
proba = float(pipeline.predict_proba(X)[0][1])
ok = pred in [0, 1] and 0.0 <= proba <= 1.0
status = "PASS" if ok else "FAIL"
print(f"  [{status}] predict={pred}  probabilidade={proba:.4f}")
sys.exit(0 if ok else 1)

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

DATA = Path("C://Users//Administrator//Downloads//logistics_delivery_dataset.csv")
df = pd.read_csv(DATA, parse_dates=["dispatch_datetime"]).sort_values("dispatch_datetime")

FEATURES = ["distance_km", "traffic_level", "weather_condition", "vehicle_type",
            "load_kg", "warehouse_region", "dispatch_hour", "day_of_week", "peak_hour"]
TARGET = "delivery_time_hours"
CAT = ["traffic_level", "weather_condition", "vehicle_type", "warehouse_region"]
NUM = [c for c in FEATURES if c not in CAT]

# Chronological split avoids training on future observations.
cut = int(len(df) * 0.80)
X_train, X_test = df[FEATURES].iloc[:cut], df[FEATURES].iloc[cut:]
y_train, y_test = df[TARGET].iloc[:cut], df[TARGET].iloc[cut:]

def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUM),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT)
    ])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=250, max_depth=14,
                                          min_samples_leaf=3, random_state=42, n_jobs=-1)
}

rows, predictions = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessing()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    predictions[name] = pred
    rows.append({"model": name,
                 "MAE_hours": mean_absolute_error(y_test, pred),
                 "RMSE_hours": mean_squared_error(y_test, pred) ** 0.5,
                 "R2": r2_score(y_test, pred)})

metrics = pd.DataFrame(rows).sort_values("RMSE_hours")

metrics.to_csv(Path(".").joinpath("model_metrics.csv"), index=False)
best_name = metrics.iloc[0]["model"]
best_pred = predictions[best_name]
pd.DataFrame({"actual_hours": y_test.to_numpy(),
              "predicted_hours": best_pred,
              "absolute_error_hours": np.abs(y_test.to_numpy() - best_pred)
             }).to_csv(Path(".").joinpath("test_predictions.csv"), index=False)

cv_estimator = Pipeline([("preprocess", preprocessing()),
                         ("model", RandomForestRegressor(n_estimators=150, max_depth=14,
                                                         min_samples_leaf=3, random_state=42,
                                                         n_jobs=-1))])
tscv = TimeSeriesSplit(n_splits=5)
cv_mae = -cross_val_score(cv_estimator, X_train, y_train, cv=tscv,
                          scoring="neg_mean_absolute_error")
print(metrics.to_string(index=False))
print(f"Best test model: {best_name}")
print(f"Time-series CV MAE: {cv_mae.mean():.3f} +/- {cv_mae.std():.3f} hours")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(y_test, best_pred, alpha=0.6, s=65)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'b--'
)

plt.xlabel("Actual delivery time (hours)")
plt.ylabel("Predicted delivery time (hours)")
plt.title(f"Actual vs predicted — {best_name}")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()
residuals = y_test.to_numpy() - best_pred

plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=25, edgecolor="black")

plt.xlabel("Residuals (hours)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.tight_layout()
plt.savefig("residual_distribution.png", dpi=300)
plt.show()


import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("data/online_retail_II.xlsx")
df = df.dropna(subset=["Customer ID", "Description"])
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df = df[df["Price"] < 100]

df["Month"] = pd.to_datetime(df["InvoiceDate"]).dt.month
df["DayOfWeek"] = pd.to_datetime(df["InvoiceDate"]).dt.dayofweek

top_countries = df["Country"].value_counts().head(10).index
df["Country_enc"] = df["Country"].apply(lambda x: x if x in top_countries else "Other")
df = pd.get_dummies(df, columns=["Country_enc"])

# CORRECTION : supprime les espaces dans les noms de colonnes
df.columns = df.columns.str.replace(" ", "_")

feature_cols = ["Quantity", "Month", "DayOfWeek"] + \
               [c for c in df.columns if c.startswith("Country_enc_")]
target_col = "Price"

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("retail_price_optimizer")

best_model_name = None
best_rmse = float("inf")
best_run_id = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):

        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "Online Retail II UCI")
        mlflow.set_tag("author", "Othman Asloun")

        mlflow.log_param("model", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_features", len(feature_cols))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name} — RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = model_name
            best_run_id = mlflow.active_run().info.run_id

print(f"\nMeilleur modèle : {best_model_name} avec RMSE = {best_rmse:.4f}")

model_uri = f"runs:/{best_run_id}/model"
registered = mlflow.register_model(model_uri=model_uri, name="retail_price_optimizer")

print(f"Modèle enregistré dans le Registry : version {registered.version}")
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.model_selection import train_test_split
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
df.columns = df.columns.str.replace(" ", "_")

feature_cols = ["Quantity", "Month", "DayOfWeek"] + \
               [c for c in df.columns if c.startswith("Country_enc_")]

X = df[feature_cols]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)
report.save_html("data/drift_report.html")

print("Rapport Evidently généré : data/drift_report.html")
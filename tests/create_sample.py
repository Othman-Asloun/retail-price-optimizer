import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 500

countries = ["United Kingdom", "France", "Germany", 
             "EIRE", "Spain", "Netherlands",
             "Belgium", "Switzerland", "Portugal", "Sweden"]

df = pd.DataFrame({
    "Invoice": [f"INV{i}" for i in range(n)],
    "StockCode": [f"STK{i%50}" for i in range(n)],
    "Description": [f"Product {i%50}" for i in range(n)],
    "Quantity": np.random.randint(1, 50, n),
    "InvoiceDate": pd.date_range("2021-01-01", periods=n, freq="h"),
    "Price": np.random.uniform(0.5, 50, n),
    "Customer ID": np.random.randint(10000, 99999, n),
    "Country": np.random.choice(countries, n)
})

os.makedirs("data", exist_ok=True)
df.to_excel("data/sample_test.xlsx", index=False)
print(f"Sample dataset created : {df.shape}")
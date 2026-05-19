# Retail Price Optimizer

Projet MLOps end-to-end de prédiction de prix optimal sur des données e-commerce réelles.  
Construit pour démontrer un pipeline ML complet : de l'analyse des données jusqu'au serving en production.

---

## Contexte business

Un retailer e-commerce veut optimiser ses prix en fonction du comportement d'achat, de la saisonnalité et de la géographie.  
Ce projet modélise l'élasticité-prix et prédit le prix optimal pour maximiser les ventes.

---

## Architecture du projet

retail-price-optimizer/
├── data/                  # Dataset (non versionné)
├── notebooks/
│   └── 01_EDA.ipynb       # Analyse exploratoire et élasticité-prix
├── src/
│   ├── train.py           # Entraînement + tracking MLflow
│   ├── evaluate.py        # Sélection et promotion en Production
│   └── monitor.py         # Monitoring data drift avec Evidently
├── api/
│   └── main.py            # API FastAPI de serving
├── requirements.txt
├── Dockerfile
└── README.md


---

##  Stack technique

| Catégorie | Outils |
|-----------|--------|
| Machine Learning | Scikit-Learn, XGBoost |
| MLOps | MLflow (tracking, registry, serving) |
| Monitoring | Evidently (data drift) |
| API | FastAPI, Uvicorn |
| Containerisation | Docker |
| Langage | Python 3.12 |

---

## Résultats

| Modèle | RMSE | MAE | R² |
|--------|------|-----|----|
| Linear Regression | 3.4595 | 2.0944 | 0.006 |
| Random Forest | 3.1381 | 1.7706 | 0.182 |
| **XGBoost** | **3.1337** | **1.7752** | **0.184** |

Le meilleur modèle (XGBoost) est automatiquement sélectionné via MLflow et promu en Production dans le Model Registry.

---

##  Lancer le projet

### 1. Cloner le repo et installer les dépendances

```bash
git clone https://github.com/TON_USERNAME/retail-price-optimizer.git
cd retail-price-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Télécharger le dataset

Télécharge le dataset **Online Retail II** depuis UCI :  
https://archive.ics.uci.edu/dataset/502/online+retail+ii  
Place le fichier dans `data/online_retail_II.xlsx`

### 3. Entraîner les modèles

```bash
python3 src/train.py
```

### 4. Promouvoir le meilleur modèle en Production

```bash
python3 src/evaluate.py
```

### 5. Lancer l'API

```bash
uvicorn api.main:app --reload
```

API disponible sur : http://localhost:8000  
Documentation Swagger : http://localhost:8000/docs

### 6. Visualiser les expériences MLflow

```bash
mlflow ui
```

Dashboard disponible sur : http://localhost:5000

### 7. Générer le rapport de monitoring

```bash
python3 src/monitor.py
```

Rapport généré dans `data/drift_report.html`

---

## Exemple de prédiction

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "Quantity": 10,
  "Month": 6,
  "DayOfWeek": 2,
  "Country_enc_France": 1
}'
```

Réponse :
```json
{
  "predicted_price": 1.91
}
```



## Auteur

**Othman Asloun**  
M1 Machine Learning for Data Science - Université Paris Cité  
othman.asloun@hotmail.com

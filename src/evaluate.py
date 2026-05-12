from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()
model_name = "retail_price_optimizer"

experiment = mlflow.get_experiment_by_name("retail_price_optimizer")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"]
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_rmse = best_run.data.metrics["rmse"]

print(f"Meilleur modèle : {best_run.data.tags['model_type']}")
print(f"RMSE : {best_rmse:.4f}")
print(f"Run ID : {best_run_id}")

versions = client.search_model_versions(f"name='{model_name}'")
best_version = next(v for v in versions if v.run_id == best_run_id)

client.transition_model_version_stage(
    name=model_name,
    version=best_version.version,
    stage="Production"
)

print(f"Modèle version {best_version.version} promu en Production avec succès")
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd


class ModelTraining:
    def __init__(self, spark):
        self.spark = spark
        self.tabla_monthly_events = "databricks_efc_2.default.monthly_events"
        self.tabla_monthly_features = "databricks_efc_2.default.monthly_features"
        self.tabla_predictions = "databricks_efc_2.default.monthly_features"
        self.tabla_metrics = "databricks_efc_2.default.metrics"
        self.tabla_train = "databricks_efc_2.default.train"
        self.tabla_test = "databricks_efc_2.default.test"
        self.model_name = "propension_abandono"
        self.model_version = -1
        self.features = ['x1','x2','x3']
        self.test_size = 0.2
        self.seed = 99

    def run(self):
        print(" ** job MODEL TRAINING ** ")

        df_events = self.spark.read.table(self.tabla_monthly_events).toPandas()
        X = df_events[self.features]
        y = df_events['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

        # mlflow (experiment)
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(f"/Users/edufer01@ucm.es/experiment_{self.model_name}")
        mlflow.sklearn.autolog()
        with mlflow.start_run() as run:
            model = LogisticRegression(random_state=self.seed)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"Score del modelo: {score:.2f}")
            mlflow.log_metric("Score de edu", score)

            # mlflow (model registry)
            y_pred = model.predict(X_test)
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn-model",
                signature=signature,
                registered_model_name=self.model_name,
            )

        client = MlflowClient()
        self.model_version = int(client.get_registered_model(self.model_name).latest_versions[-1].version)
        metrics = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": run.info.run_id,
            "score": score,
        }
        print(f"Creada la version {self.model_version} del modelo {self.model_name}")

        # save metrics table
        pdf_metrics = pd.DataFrame([metrics])
        df = self.spark.createDataFrame(pdf_metrics)
        df.write.mode("append").saveAsTable(self.tabla_metrics)

        # save train table
        df_train = pd.merge(X_train, y_train, left_index=True, right_index=True, how='inner')
        df_train['model_name'] = self.model_name
        df_train['model_version'] = self.model_version
        sdf = self.spark.createDataFrame(df_train)
        sdf.write.mode("append").saveAsTable(self.tabla_train)

        # save test table
        df_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='inner')
        df_test['model_name'] = self.model_name
        df_test['model_version'] = self.model_version
        sdf = self.spark.createDataFrame(df_test)
        sdf.write.mode("append").saveAsTable(self.tabla_test)

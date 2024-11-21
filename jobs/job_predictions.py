import mlflow
from mlflow import MlflowClient
import pandas as pd
from datetime import datetime


class ModelPredictions:
    def __init__(self, spark):
        self.spark = spark
        self.tabla_features = "databricks_efc_2.default.monthly_features"
        self.tabla_predictions = "databricks_efc_2.default.monthly_predictions"
        self.model_name = "propension_abandono"
        self.model_version = -1
        self.features = ['x1','x2','x3']

    def run(self):
        print(" ** job MAKE PREDICTIONS ** ")

        pdf_features = (
            self.spark.read.table(self.tabla_features)
            .toPandas()
        )

        mlflow.set_tracking_uri("databricks")
        client = MlflowClient()
        self.model_version = int(client.get_registered_model(self.model_name).latest_versions[-1].version)
        model_uri = f"models:/{self.model_name}/{str(self.model_version)}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Cargamos el modelo {self.model_name} con versión {self.model_version} para hacer las predicciones")

        y_hat = model.predict(pdf_features[self.features])

        # save predictions table
        clientes = pdf_features['id_cliente'].to_list()
        n = len(clientes)
        predictions = {
            "id_cliente": clientes,
            "y_hat": y_hat,
            "model_name": [self.model_name] * n,
            "model_version": [self.model_version] * n,
        }
        df_predictions = pd.DataFrame(predictions)
        sdf = self.spark.createDataFrame(df_predictions)
        sdf.write.mode("append").saveAsTable(self.tabla_predictions)

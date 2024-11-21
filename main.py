from databricks.connect import DatabricksSession as SparkSession
from databricks.sdk.core import Config
from jobs.job_training import ModelTraining


if __name__ == "__main__":
    config = Config(profile="DEFAULT", cluster_id="1120-124406-2os3k2iw")
    spark = SparkSession.builder.sdkConfig(config).getOrCreate()

    # df = spark.read.table("databricks_efc_2.default.monthly_events")
    # df.show(5)

    job = ModelTraining(spark)
    job.run()



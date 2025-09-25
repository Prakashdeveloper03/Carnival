# run_iceberg_demo.py
import seaborn as sns
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("IcebergNessieScript").getOrCreate()
    print("✅ Spark Session Created!")

    # --- 1. Load Data and Create Table ---
    print("\n--- 1. Loading Iris dataset and creating Iceberg table ---")
    iris_df = sns.load_dataset("iris")
    spark_iris_df = spark.createDataFrame(iris_df)

    spark.sql("CREATE NAMESPACE IF NOT EXISTS nessie.iris_db")
    spark.sql("DROP TABLE IF EXISTS nessie.iris_db.iris_table")
    spark.sql(
        """
        CREATE TABLE nessie.iris_db.iris_table (
            sepal_length DOUBLE, sepal_width DOUBLE,
            petal_length DOUBLE, petal_width DOUBLE, species STRING
        ) USING iceberg PARTITIONED BY (species)
    """
    )
    spark_iris_df.writeTo("nessie.iris_db.iris_table").append()
    print("Table 'iris_table' created and populated.")
    spark.sql("SELECT count(*) as count FROM nessie.iris_db.iris_table").show()

    # --- 2. Time Travel 🕰️ ---
    print("\n--- 2. Demonstrating Time Travel ---")
    history_df = spark.sql("SELECT snapshot_id FROM nessie.iris_db.iris_table.history")
    initial_snapshot_id = history_df.first()["snapshot_id"]

    print(f"Initial snapshot ID: {initial_snapshot_id}")
    spark.sql(
        "UPDATE nessie.iris_db.iris_table SET sepal_length = 5.55 WHERE species = 'setosa' AND sepal_width = 3.5"
    )

    print("Row updated. Querying current state:")
    spark.sql(
        "SELECT * FROM nessie.iris_db.iris_table WHERE sepal_length = 5.55"
    ).show()

    print("Traveling back in time to query BEFORE the update:")
    spark.sql(
        f"SELECT * FROM nessie.iris_db.iris_table VERSION AS OF {initial_snapshot_id} WHERE species = 'setosa' AND sepal_width = 3.5"
    ).show()

    # --- 3. Schema Evolution 🧬 ---
    print("\n--- 3. Demonstrating Schema Evolution ---")
    spark.sql("ALTER TABLE nessie.iris_db.iris_table ADD COLUMN source STRING")
    spark.sql(
        "ALTER TABLE nessie.iris_db.iris_table RENAME COLUMN sepal_width TO sepal_width_cm"
    )
    spark.sql("ALTER TABLE nessie.iris_db.iris_table DROP COLUMN petal_width")
    print("Schema evolved. New schema:")
    spark.sql("SELECT * FROM nessie.iris_db.iris_table LIMIT 5").show()

    # --- 4. MERGE INTO Operation 🔄 ---
    print("\n--- 4. Demonstrating MERGE INTO ---")
    update_data = [
        (7.0, 3.2, 4.7, "virginica", "new_batch"),
        (5.0, 2.5, 3.0, "new_hybrid", "new_batch"),
    ]
    update_df = spark.createDataFrame(
        update_data,
        ["sepal_length", "sepal_width_cm", "petal_length", "species", "source"],
    )
    update_df.createOrReplaceTempView("iris_updates")
    spark.sql(
        """
        MERGE INTO nessie.iris_db.iris_table t
        USING iris_updates u ON t.species = u.species AND t.petal_length = u.petal_length
        WHEN MATCHED THEN UPDATE SET t.sepal_length = u.sepal_length, t.source = u.source
        WHEN NOT MATCHED THEN INSERT *
    """
    )
    print("MERGE complete. Verifying changes:")
    spark.sql(
        "SELECT * FROM nessie.iris_db.iris_table WHERE source = 'new_batch'"
    ).show()

    # --- 5. Nessie Branching & Merging 🌿 ---
    print("\n--- 5. Demonstrating Nessie Branching ---")
    spark.sql("CREATE BRANCH etl_dev IN nessie")
    spark.sql("USE REFERENCE etl_dev IN nessie")
    spark.sql("DELETE FROM nessie.iris_db.iris_table WHERE species = 'setosa'")
    print(
        "Deleted 'setosa' on 'etl_dev' branch. Species count:",
        spark.sql(
            "SELECT count(DISTINCT species) FROM nessie.iris_db.iris_table"
        ).first()[0],
    )
    spark.sql("USE REFERENCE main IN nessie")
    print(
        "Switched to 'main' branch. Species count:",
        spark.sql(
            "SELECT count(DISTINCT species) FROM nessie.iris_db.iris_table"
        ).first()[0],
    )
    spark.sql("MERGE BRANCH etl_dev INTO main IN nessie")
    print(
        "Merged 'etl_dev' into 'main'. Species count on main:",
        spark.sql(
            "SELECT count(DISTINCT species) FROM nessie.iris_db.iris_table"
        ).first()[0],
    )
    print("\n✅ Demo finished successfully!")
    spark.stop()


if __name__ == "__main__":
    main()

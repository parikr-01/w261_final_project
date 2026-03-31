# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Departure Delays: All Models Evaluation
# MAGIC
# MAGIC This notebook applies Random Forest, Lasso, Ridge, and Gradient Boosted Tree regression to the flight delay prediction task, implementing it as a regression problem with `DEP_DELAY` as the target variable. The joined OTPW-style dataset is used to construct features available prior to scheduled departure, apply a chronological train/validation/test split, tune model hyperparameters on the validation window, and compare final model performance on a held-out test set.
# MAGIC
# MAGIC Data loading, feature engineering, and model training are driven by shared functions in `modeling_utils.py`, ensuring consistency across all notebooks.
# MAGIC
# MAGIC The shared metrics reported in this notebook are `RMSE`, `MAE`, `R2`, `OTPA`, `SDDR`, and `F1`.

# COMMAND ----------

import sys
import importlib.util

if "modeling_utils" in sys.modules:
    del sys.modules["modeling_utils"]

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location(
    "modeling_utils",
    "/Workspace/Users/andrewjlei31@berkeley.edu/w_261_final_project/modeling_utils.py"
)
modeling_utils = importlib.util.module_from_spec(spec)
sys.modules["modeling_utils"] = modeling_utils
spec.loader.exec_module(modeling_utils)

# COMMAND ----------

import pandas as pd
from modeling_utils import (
    DEFAULT_DELAY_THRESHOLD,
    DEFAULT_SEVERE_DELAY_THRESHOLD,
    evaluate_model,
    fit_gbt_model,
    fit_linear_model,
    fit_rf_model,
    fit_stacking_ensemble,
    load_and_prepare_data,
    prune_empty_feature_columns,
    run_gbt_search_and_eval,
    run_linear_search_and_eval,
    search_rf_model,
    top_gbt_importances,
    top_linear_coefficients,
    top_rf_importances,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
DATA_PATH = f"{data_BASE_DIR}datasets_final_project_2022/parquet_airlines_data_1y/"
ROW_FILTER = "YEAR = 2019 AND QUARTER = 4"
DATE_COL = "FL_DATE"
TARGET_COL = "DEP_DELAY"

TRAIN_END = "2019-11-15"
VALID_END = "2019-11-30"
AUTO_INFER_SPLITS = False
SEED = 0

DELAY_THRESHOLD = DEFAULT_DELAY_THRESHOLD
SEVERE_DELAY_THRESHOLD = DEFAULT_SEVERE_DELAY_THRESHOLD

RF_PARAM_GRID = [
    {"numTrees": 50,  "maxDepth": 5},
    {"numTrees": 100, "maxDepth": 10},
    {"numTrees": 100, "maxDepth": 15},
    {"numTrees": 200, "maxDepth": 10},
    {"numTrees": 200, "maxDepth": 20},
    {"numTrees": 200, "maxDepth": 30},
]

LASSO_REG_PARAMS = [0.001, 0.01, 0.05]
RIDGE_REG_PARAMS = [0.01, 0.10, 1.00]
LINEAR_MAX_ITER = 100

GBT_PARAM_GRID = [
    {"maxDepth": 5, "maxIter": 40, "stepSize": 0.05},
    {"maxDepth": 7, "maxIter": 60, "stepSize": 0.05},
    {"maxDepth": 5, "maxIter": 80, "stepSize": 0.10},
]

# COMMAND ----------

# start Spark Session (RUN THIS CELL AS IS)
#Step A: Start Spark by running the following cell

import os
import sys
os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17'

import pyspark
from pyspark.sql import SparkSession

# Clean up stale 'spark' module if it was imported by mistake earlier
if 'spark' in sys.modules and not isinstance(sys.modules['spark'], SparkSession):
    del sys.modules['spark']
    try:
        del spark
    except NameError:
        pass

try:
    spark
    print('Spark is already running')
    sc = spark.sparkContext
    print(f'{sc.master} appName: {sc.appName}')
except NameError:
    print('starting Spark')
    app_name = 'random_forest'
    master = 'local[*]'
    spark = SparkSession\
            .builder\
            .appName(app_name)\
            .master(master)\
            .getOrCreate()

# Don't worry about messages shown below

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion and Feature Engineering
# MAGIC
# MAGIC Data is loaded from the OTPW joined dataset, feature-engineered (weather flags, scheduled departure hour, weekend indicator, numeric parsing), and split chronologically into train/validation/test sets — all via `load_and_prepare_data()` from `modeling_utils`.

# COMMAND ----------

data = load_and_prepare_data(
    spark,
    data_path=DATA_PATH,
    row_filter=ROW_FILTER,
    target_col=TARGET_COL,
    date_col=DATE_COL,
    train_end=TRAIN_END,
    valid_end=VALID_END,
    auto_infer_splits=AUTO_INFER_SPLITS,
    seed=SEED,
)

before_count = train_df.count() + valid_df.count() + test_df.count()
train_df = train_df.dropna()
valid_df = valid_df.dropna()
test_df = test_df.dropna()
after_count = train_df.count() + valid_df.count() + test_df.count()

print(before_count)
print(after_count)

model_df = data["model_df"]
train_df = data["train_df"]
valid_df = data["valid_df"]
test_df = data["test_df"]
split_summary = data["split_summary"]
numeric_cols = data["numeric_cols"]
linear_cat_cols = data["linear_cat_cols"]
tree_cat_cols = data["tree_cat_cols"]

print("Split summary")
print(split_summary.to_string(index=False))
print("\nNumeric features:", numeric_cols)
print("Linear categorical features:", linear_cat_cols)
print("Tree categorical features:", tree_cat_cols)

split_summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lasso and Ridge Training
# MAGIC
# MAGIC Lasso and Ridge regularized linear models are searched and evaluated via `run_linear_search_and_eval()` from `modeling_utils`, using the same data prepared above.

# COMMAND ----------

linear_results = run_linear_search_and_eval(
    train_df=train_df,
    valid_df=valid_df,
    test_df=test_df,
    numeric_cols=numeric_cols,
    linear_cat_cols=linear_cat_cols,
    lasso_reg_params=LASSO_REG_PARAMS,
    ridge_reg_params=RIDGE_REG_PARAMS,
    linear_max_iter=LINEAR_MAX_ITER,
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
)

print("Lasso validation search:")
display(linear_results["lasso_search"]) if "display" in dir() else print(linear_results["lasso_search"])
print("\nRidge validation search:")
display(linear_results["ridge_search"]) if "display" in dir() else print(linear_results["ridge_search"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient Boosted Trees Training
# MAGIC
# MAGIC GBT is searched and evaluated via `run_gbt_search_and_eval()` from `modeling_utils`, reusing the `train_valid_df` already computed by the linear step.

# COMMAND ----------

gbt_results = run_gbt_search_and_eval(
    train_df=train_df,
    valid_df=valid_df,
    test_df=test_df,
    numeric_cols=numeric_cols,
    tree_cat_cols=tree_cat_cols,
    gbt_param_grid=GBT_PARAM_GRID,
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
    seed=SEED,
    train_valid_df=linear_results["train_valid_df"],
)

print("GBT validation search:")
display(gbt_results["gbt_search"]) if "display" in dir() else print(gbt_results["gbt_search"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Training and Validation
# MAGIC
# MAGIC Random Forest is trained on the same data splits. Hyperparameters are searched on the validation set, then the best configuration is refit on train+validation for final test evaluation.

# COMMAND ----------

rf_numeric_cols, rf_cat_cols = prune_empty_feature_columns(train_df, numeric_cols, tree_cat_cols)

best_rf, rf_search = search_rf_model(
    train_df=train_df,
    valid_df=valid_df,
    numeric_columns=rf_numeric_cols,
    categorical_columns=rf_cat_cols,
    param_grid=RF_PARAM_GRID,
    model_name="RandomForest",
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
    seed=SEED,
)

print("RF validation search results:")
rf_search

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Model Evaluation
# MAGIC
# MAGIC After selecting the best validation configuration for each model, the training and validation windows are combined and the Random Forest is refit on the enlarged development set. Final performance for all four models (Lasso, Ridge, GBT, Random Forest) is compared on the held-out test set.

# COMMAND ----------

train_valid_df = linear_results["train_valid_df"]

rf_model = fit_rf_model(
    train_df=train_valid_df,
    numeric_columns=rf_numeric_cols,
    categorical_columns=rf_cat_cols,
    num_trees=int(best_rf["numTrees"]),
    max_depth=int(best_rf["maxDepth"]),
    max_bins=int(best_rf["maxBins"]),
    subsampling_rate=float(best_rf["subsamplingRate"]),
    seed=SEED,
)

test_results = pd.DataFrame(
    [
        linear_results["lasso_test"],
        linear_results["ridge_test"],
        gbt_results["gbt_test"],
        {
            **evaluate_model(rf_model, test_df, model_name="RandomForest", split_name="test",
                             delay_threshold=DELAY_THRESHOLD, severe_delay_threshold=SEVERE_DELAY_THRESHOLD),
            "numTrees": best_rf["numTrees"],
            "maxDepth": best_rf["maxDepth"],
            "maxBins": best_rf["maxBins"],
            "subsamplingRate": best_rf["subsamplingRate"],
        },
    ]
).sort_values(["RMSE", "model"]).reset_index(drop=True)

test_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Interpretation
# MAGIC
# MAGIC Top learned signals for each model: largest absolute coefficients for Lasso/Ridge, ranked feature importances for GBT and Random Forest.

# COMMAND ----------

lasso_features = top_linear_coefficients(linear_results["lasso_model"], train_valid_df, top_n=20)
ridge_features = top_linear_coefficients(linear_results["ridge_model"], train_valid_df, top_n=20)
gbt_features = top_gbt_importances(gbt_results["gbt_model"], train_valid_df, top_n=20)
rf_features = top_rf_importances(rf_model, train_valid_df, top_n=20)

for name, table in [("Lasso", lasso_features), ("Ridge", ridge_features),
                     ("GBT", gbt_features), ("Random Forest", rf_features)]:
    print(f"\nTop signals for {name}\n")
    if "display" in globals():
        display(table)
    else:
        print(table.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stacking Ensemble
# MAGIC
# MAGIC A Ridge-regularized linear regression meta-learner is trained on the validation-set predictions of all four base models. The base models used for validation predictions are trained on `train_df` only, so the meta-learner sees out-of-sample predictions. For final test evaluation, the base models retrained on `train+valid` (already fitted above) generate test predictions, which the meta-learner combines into a single stacked prediction.

# COMMAND ----------

# Refit base models on train_df only (out-of-sample for validation predictions)
lasso_train_only = fit_linear_model(
    train_df=train_df,
    numeric_columns=linear_results["linear_numeric_cols"],
    categorical_columns=linear_results["linear_cat_cols"],
    reg_param=linear_results["best_lasso"]["regParam"],
    elastic_net_param=linear_results["best_lasso"]["elasticNetParam"],
    max_iter=int(linear_results["best_lasso"]["maxIter"]),
)

ridge_train_only = fit_linear_model(
    train_df=train_df,
    numeric_columns=linear_results["linear_numeric_cols"],
    categorical_columns=linear_results["linear_cat_cols"],
    reg_param=linear_results["best_ridge"]["regParam"],
    elastic_net_param=linear_results["best_ridge"]["elasticNetParam"],
    max_iter=int(linear_results["best_ridge"]["maxIter"]),
)

gbt_train_only = fit_gbt_model(
    train_df=train_df,
    numeric_columns=gbt_results["gbt_numeric_cols"],
    categorical_columns=gbt_results["gbt_cat_cols"],
    max_depth=int(gbt_results["best_gbt"]["maxDepth"]),
    max_iter=int(gbt_results["best_gbt"]["maxIter"]),
    step_size=float(gbt_results["best_gbt"]["stepSize"]),
    seed=SEED,
)

rf_train_only = fit_rf_model(
    train_df=train_df,
    numeric_columns=rf_numeric_cols,
    categorical_columns=rf_cat_cols,
    num_trees=int(best_rf["numTrees"]),
    max_depth=int(best_rf["maxDepth"]),
    max_bins=int(best_rf["maxBins"]),
    subsampling_rate=float(best_rf["subsamplingRate"]),
    seed=SEED,
)


# COMMAND ----------

base_models_train = {
    "Lasso": lasso_train_only,
    "Ridge": ridge_train_only,
    "GBT": gbt_train_only,
    "RF": rf_train_only,
}

base_models_final = {
    "Lasso": linear_results["lasso_model"],
    "Ridge": linear_results["ridge_model"],
    "GBT": gbt_results["gbt_model"],
    "RF": rf_model,
}

stacking = fit_stacking_ensemble(
    base_models_train=base_models_train,
    base_models_final=base_models_final,
    valid_df=valid_df,
    test_df=test_df,
    meta_reg_param=0.01,
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
)

print("Meta-learner weights (Ridge regression on base model predictions):")
print(stacking["meta_weights"].to_string(index=False))
print(f"\nMeta-learner intercept: {stacking['meta_intercept']:.4f}")

# COMMAND ----------

train_results = pd.concat(
    pd.DataFrame([stacking["stacked_train_metrics"]]),
    ignore_index=True,
    sort=False,
).sort_values(["RMSE", "model"]).reset_index(drop=True)


# COMMAND ----------

train_results = pd.concat(
    [train_resul;ts, pd.DataFrame([stacking["stacked_test_metrics"]])],
    ignore_index=True,
    sort=False,
).sort_values(["RMSE", "model"]).reset_index(drop=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Combined Test Results (All Models + Stacking)

# COMMAND ----------

all_test_results = pd.concat(
    [test_results, pd.DataFrame([stacking["stacked_test_metrics"]])],
    ignore_index=True,
    sort=False,
).sort_values(["RMSE", "model"]).reset_index(drop=True)

all_test_results

# COMMAND ----------


# Databricks notebook source
# MAGIC %md
# MAGIC # Flight Delay Prediction — Full Pipeline
# MAGIC
# MAGIC End-to-end pipeline integrating work from all team members:
# MAGIC
# MAGIC | Stage | Description |
# MAGIC |-------|-------------|
# MAGIC | **Data ingestion** | Load OTPW joined dataset (flight + weather + station lookup) |
# MAGIC | **Data validation** | Null rates, value ranges, duplicate checks |
# MAGIC | **Data cleaning** | Drop cancelled/diverted, impute nulls, cast timestamps |
# MAGIC | **Feature engineering** | Cyclical departure time, route avg delay, weather severity score |
# MAGIC | **Feature preprocessing** | OHE categoricals, StandardScaler numerics, VectorAssembler |
# MAGIC | **Model training** | Lasso, Ridge, Gradient Boosted Trees, Random Forest |
# MAGIC | **Stacking ensemble** | Ridge regression meta-learner on out-of-fold predictions |
# MAGIC | **Evaluation** | RMSE · MAE · R² · OTPA · SDDR · F1 |

# COMMAND ----------

import sys
import importlib.util

if "modeling_utils" in sys.modules:
    del sys.modules["modeling_utils"]

sys.dont_write_bytecode = True

# Databricks workspace import — update this path to match your workspace location
spec = importlib.util.spec_from_file_location(
    "modeling_utils",
    "/Workspace/Users/andrewjlei31@berkeley.edu/w_261_final_project/modeling_utils.py",
)
modeling_utils = importlib.util.module_from_spec(spec)
sys.modules["modeling_utils"] = modeling_utils
spec.loader.exec_module(modeling_utils)

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

from modeling_utils import (
    DEFAULT_DELAY_THRESHOLD,
    DEFAULT_SEVERE_DELAY_THRESHOLD,
    add_route_avg_delay,
    compute_route_avg_delay,
    describe_splits,
    evaluate_model,
    fit_gbt_model,
    fit_linear_model,
    fit_rf_model,
    fit_stacking_ensemble,
    infer_time_split_boundaries,
    prepare_modeling_frame,
    prune_empty_feature_columns,
    resolve_feature_columns,
    search_gbt_model,
    search_linear_model,
    search_rf_model,
    time_based_split,
    top_gbt_importances,
    top_linear_coefficients,
    top_rf_importances,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
DATA_PATH = f"{data_BASE_DIR}OTPW_60M/OTPW_60M/"

ROW_FILTER = "YEAR = 2019 AND QUARTER = 4"
DATE_COL = "FL_DATE"
TARGET_COL = "DEP_DELAY"

TRAIN_END = "2019-11-15"
VALID_END = "2019-11-30"
AUTO_INFER_SPLITS = False
SEED = 42

DELAY_THRESHOLD = DEFAULT_DELAY_THRESHOLD        # 15 min
SEVERE_DELAY_THRESHOLD = DEFAULT_SEVERE_DELAY_THRESHOLD  # 60 min

# --- Hyperparameter grids ---

LASSO_REG_PARAMS = [0.001, 0.01, 0.05]
RIDGE_REG_PARAMS = [0.01, 0.10, 1.00]
LINEAR_MAX_ITER = 100

GBT_PARAM_GRID = [
    {"maxDepth": 5, "maxIter": 40, "stepSize": 0.05},
    {"maxDepth": 7, "maxIter": 60, "stepSize": 0.05},
    {"maxDepth": 5, "maxIter": 80, "stepSize": 0.10},
]

RF_PARAM_GRID = [
    {"numTrees": 50,  "maxDepth": 5},
    {"numTrees": 100, "maxDepth": 10},
    {"numTrees": 100, "maxDepth": 15},
    {"numTrees": 200, "maxDepth": 10},
    {"numTrees": 200, "maxDepth": 20},
    {"numTrees": 200, "maxDepth": 30},
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Data Ingestion
# MAGIC
# MAGIC Load the pre-joined OTPW dataset (flight records + weather observations + station lookup),
# MAGIC filtered to the analysis window.

# COMMAND ----------

raw_df = spark.read.parquet(DATA_PATH)

if ROW_FILTER:
    raw_df = raw_df.filter(ROW_FILTER)

print(f"Raw rows: {raw_df.count():,}")
print(f"Raw columns: {len(raw_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Data Validation
# MAGIC
# MAGIC Quick quality checks before cleaning: null rates for critical columns,
# MAGIC HourlyWindSpeed range, and FL_DATE + TAIL_NUM duplicate flights.

# COMMAND ----------

total_rows = raw_df.count()

validation_cols = [TARGET_COL, DATE_COL, "HourlyWindSpeed", "CANCELLED", "TAIL_NUM"]
validation_cols = [c for c in validation_cols if c in raw_df.columns]

null_counts = raw_df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in validation_cols
]).collect()[0].asDict()

print("=== Null counts ===")
for col_name, cnt in null_counts.items():
    pct = cnt / total_rows * 100 if total_rows > 0 else 0
    print(f"  {col_name}: {cnt:,} ({pct:.1f}%)")

if "HourlyWindSpeed" in raw_df.columns:
    ws = raw_df.select(
        F.expr("try_cast(HourlyWindSpeed as double)").alias("ws")
    ).where(F.col("ws").isNotNull())
    ws_stats = ws.agg(F.min("ws"), F.max("ws"), F.mean("ws")).collect()[0]
    print(f"\n=== HourlyWindSpeed range === min={ws_stats[0]}, max={ws_stats[1]}, mean={ws_stats[2]:.1f}")

if "TAIL_NUM" in raw_df.columns and DATE_COL in raw_df.columns:
    dupes = (
        raw_df.groupBy(DATE_COL, "TAIL_NUM")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    print(f"\n=== Duplicate FL_DATE + TAIL_NUM combinations === {dupes:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Data Cleaning & Feature Engineering
# MAGIC
# MAGIC `prepare_modeling_frame` handles:
# MAGIC - Drop cancelled & diverted flights
# MAGIC - Parse numeric weather strings (HourlyPrecipitation, etc.) and impute trace values
# MAGIC - Cast timestamps and compute **scheduled departure hour**
# MAGIC - **Cyclical encoding** of departure hour (sin/cos)
# MAGIC - **Weather severity score** (composite of rain/snow/fog/thunder flags)
# MAGIC - Weekend indicator, day-of-week, month, quarter

# COMMAND ----------

model_df = prepare_modeling_frame(
    raw_df,
    target_col=TARGET_COL,
    date_col=DATE_COL,
    seed=SEED,
)

numeric_cols, linear_cat_cols, tree_cat_cols = resolve_feature_columns(model_df)

print(f"Model-ready rows: {model_df.count():,}")
print(f"Model-ready columns: {len(model_df.columns)}")
print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Linear categorical features ({len(linear_cat_cols)}): {linear_cat_cols}")
print(f"Tree categorical features ({len(tree_cat_cols)}): {tree_cat_cols}")

model_df.limit(5).display() if "display" in dir() else model_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Chronological Train / Validation / Test Split
# MAGIC
# MAGIC Temporal ordering prevents leakage from future observations into past predictions.

# COMMAND ----------

if AUTO_INFER_SPLITS or TRAIN_END is None or VALID_END is None:
    TRAIN_END, VALID_END = infer_time_split_boundaries(model_df, date_col=DATE_COL)

train_df, valid_df, test_df = time_based_split(
    model_df, date_col=DATE_COL, train_end=TRAIN_END, valid_end=VALID_END,
)

split_summary = describe_splits(train_df, valid_df, test_df, date_col=DATE_COL)
if (split_summary["rows"] == 0).any():
    raise ValueError("At least one split is empty — adjust ROW_FILTER or split boundaries.")

print("Split summary")
print(split_summary.to_string(index=False))
split_summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Route Average Delay Feature
# MAGIC
# MAGIC Historical average departure delay per ORIGIN→DEST route, computed from
# MAGIC **training data only** to prevent leakage, then joined to all three splits.

# COMMAND ----------

route_avgs = compute_route_avg_delay(train_df)

train_df = add_route_avg_delay(train_df, route_avgs)
valid_df = add_route_avg_delay(valid_df, route_avgs)
test_df = add_route_avg_delay(test_df, route_avgs)

numeric_cols = numeric_cols + ["route_avg_delay"]
print(f"Numeric features after route avg delay ({len(numeric_cols)}): {numeric_cols}")

# COMMAND ----------

# Drop remaining nulls and cache
before_count = train_df.count() + valid_df.count() + test_df.count()

train_df = train_df.dropna().cache()
valid_df = valid_df.dropna().cache()
test_df = test_df.dropna().cache()

after_count = train_df.count() + valid_df.count() + test_df.count()
print(f"Rows before dropna: {before_count:,}")
print(f"Rows after dropna:  {after_count:,}")
print(f"Dropped: {before_count - after_count:,} ({(before_count - after_count) / before_count * 100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Feature Preprocessing & Column Pruning
# MAGIC
# MAGIC Remove any feature columns that are entirely empty after the split + dropna,
# MAGIC separately for linear models (which use OHE + StandardScaler) and tree models.

# COMMAND ----------

linear_numeric_cols, pruned_linear_cat_cols = prune_empty_feature_columns(
    train_df, numeric_cols, linear_cat_cols,
)
tree_numeric_cols, pruned_tree_cat_cols = prune_empty_feature_columns(
    train_df, numeric_cols, tree_cat_cols,
)

print(f"Linear — numeric: {linear_numeric_cols}")
print(f"Linear — categorical: {pruned_linear_cat_cols}")
print(f"Tree   — numeric: {tree_numeric_cols}")
print(f"Tree   — categorical: {pruned_tree_cat_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Lasso & Ridge Regression
# MAGIC
# MAGIC Regularized linear baselines. Lasso (L1) for feature selection, Ridge (L2) for
# MAGIC handling collinear weather features. Hyperparameters are searched on the validation set.

# COMMAND ----------

best_lasso, lasso_search = search_linear_model(
    train_df=train_df,
    valid_df=valid_df,
    numeric_columns=linear_numeric_cols,
    categorical_columns=pruned_linear_cat_cols,
    reg_params=LASSO_REG_PARAMS,
    elastic_net_param=1.0,
    model_name="Lasso",
    max_iter=LINEAR_MAX_ITER,
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
)

best_ridge, ridge_search = search_linear_model(
    train_df=train_df,
    valid_df=valid_df,
    numeric_columns=linear_numeric_cols,
    categorical_columns=pruned_linear_cat_cols,
    reg_params=RIDGE_REG_PARAMS,
    elastic_net_param=0.0,
    model_name="Ridge",
    max_iter=LINEAR_MAX_ITER,
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
)

print("Lasso validation search:")
lasso_search

# COMMAND ----------

print("Ridge validation search:")
ridge_search

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Gradient Boosted Trees
# MAGIC
# MAGIC Non-linear model to capture threshold-based and interaction effects identified in EDA.

# COMMAND ----------

best_gbt, gbt_search = search_gbt_model(
    train_df=train_df,
    valid_df=valid_df,
    numeric_columns=tree_numeric_cols,
    categorical_columns=pruned_tree_cat_cols,
    param_grid=GBT_PARAM_GRID,
    model_name="GBT",
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
    seed=SEED,
)

print("GBT validation search:")
gbt_search

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Random Forest
# MAGIC
# MAGIC Ensemble tree model providing complementary diversity to GBT for the stacking layer.

# COMMAND ----------

best_rf, rf_search = search_rf_model(
    train_df=train_df,
    valid_df=valid_df,
    numeric_columns=tree_numeric_cols,
    categorical_columns=pruned_tree_cat_cols,
    param_grid=RF_PARAM_GRID,
    model_name="RandomForest",
    delay_threshold=DELAY_THRESHOLD,
    severe_delay_threshold=SEVERE_DELAY_THRESHOLD,
    seed=SEED,
)

print("Random Forest validation search:")
rf_search

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — Final Model Evaluation (All Base Models)
# MAGIC
# MAGIC Refit each model on `train + validation`, then evaluate on the held-out test set.

# COMMAND ----------

train_valid_df = train_df.unionByName(valid_df).cache()

lasso_model = fit_linear_model(
    train_df=train_valid_df,
    numeric_columns=linear_numeric_cols,
    categorical_columns=pruned_linear_cat_cols,
    reg_param=best_lasso["regParam"],
    elastic_net_param=best_lasso["elasticNetParam"],
    max_iter=int(best_lasso["maxIter"]),
)

ridge_model = fit_linear_model(
    train_df=train_valid_df,
    numeric_columns=linear_numeric_cols,
    categorical_columns=pruned_linear_cat_cols,
    reg_param=best_ridge["regParam"],
    elastic_net_param=best_ridge["elasticNetParam"],
    max_iter=int(best_ridge["maxIter"]),
)

gbt_model = fit_gbt_model(
    train_df=train_valid_df,
    numeric_columns=tree_numeric_cols,
    categorical_columns=pruned_tree_cat_cols,
    max_depth=int(best_gbt["maxDepth"]),
    max_iter=int(best_gbt["maxIter"]),
    step_size=float(best_gbt["stepSize"]),
    seed=SEED,
)

rf_model = fit_rf_model(
    train_df=train_valid_df,
    numeric_columns=tree_numeric_cols,
    categorical_columns=pruned_tree_cat_cols,
    num_trees=int(best_rf["numTrees"]),
    max_depth=int(best_rf["maxDepth"]),
    max_bins=int(best_rf["maxBins"]),
    subsampling_rate=float(best_rf["subsamplingRate"]),
    seed=SEED,
)

# COMMAND ----------

test_results = pd.DataFrame([
    {
        **evaluate_model(lasso_model, test_df, model_name="Lasso", split_name="test",
                         delay_threshold=DELAY_THRESHOLD, severe_delay_threshold=SEVERE_DELAY_THRESHOLD),
        "regParam": best_lasso["regParam"],
        "elasticNetParam": best_lasso["elasticNetParam"],
    },
    {
        **evaluate_model(ridge_model, test_df, model_name="Ridge", split_name="test",
                         delay_threshold=DELAY_THRESHOLD, severe_delay_threshold=SEVERE_DELAY_THRESHOLD),
        "regParam": best_ridge["regParam"],
        "elasticNetParam": best_ridge["elasticNetParam"],
    },
    {
        **evaluate_model(gbt_model, test_df, model_name="GBT", split_name="test",
                         delay_threshold=DELAY_THRESHOLD, severe_delay_threshold=SEVERE_DELAY_THRESHOLD),
        "maxDepth": best_gbt["maxDepth"],
        "maxIter": best_gbt["maxIter"],
        "stepSize": best_gbt["stepSize"],
    },
    {
        **evaluate_model(rf_model, test_df, model_name="RandomForest", split_name="test",
                         delay_threshold=DELAY_THRESHOLD, severe_delay_threshold=SEVERE_DELAY_THRESHOLD),
        "numTrees": best_rf["numTrees"],
        "maxDepth": best_rf["maxDepth"],
    },
]).sort_values(["RMSE", "model"]).reset_index(drop=True)

print("Base model test results:")
test_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11 — Model Interpretation
# MAGIC
# MAGIC Top learned signals: largest absolute coefficients for Lasso/Ridge,
# MAGIC ranked feature importances for GBT and Random Forest.

# COMMAND ----------

lasso_features = top_linear_coefficients(lasso_model, train_valid_df, top_n=20)
ridge_features = top_linear_coefficients(ridge_model, train_valid_df, top_n=20)
gbt_features = top_gbt_importances(gbt_model, train_valid_df, top_n=20)
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
# MAGIC ## Step 12 — Stacking Ensemble
# MAGIC
# MAGIC A Ridge-regularized linear regression meta-learner is trained on the
# MAGIC **validation-set predictions** of all four base models (trained on `train_df` only),
# MAGIC so the meta-learner sees out-of-sample predictions. For final test evaluation,
# MAGIC the base models retrained on `train + valid` generate test predictions which the
# MAGIC meta-learner combines into a single stacked prediction.

# COMMAND ----------

# Refit base models on train_df only (for out-of-sample validation predictions)
lasso_train_only = fit_linear_model(
    train_df=train_df,
    numeric_columns=linear_numeric_cols,
    categorical_columns=pruned_linear_cat_cols,
    reg_param=best_lasso["regParam"],
    elastic_net_param=best_lasso["elasticNetParam"],
    max_iter=int(best_lasso["maxIter"]),
)

ridge_train_only = fit_linear_model(
    train_df=train_df,
    numeric_columns=linear_numeric_cols,
    categorical_columns=pruned_linear_cat_cols,
    reg_param=best_ridge["regParam"],
    elastic_net_param=best_ridge["elasticNetParam"],
    max_iter=int(best_ridge["maxIter"]),
)

gbt_train_only = fit_gbt_model(
    train_df=train_df,
    numeric_columns=tree_numeric_cols,
    categorical_columns=pruned_tree_cat_cols,
    max_depth=int(best_gbt["maxDepth"]),
    max_iter=int(best_gbt["maxIter"]),
    step_size=float(best_gbt["stepSize"]),
    seed=SEED,
)

rf_train_only = fit_rf_model(
    train_df=train_df,
    numeric_columns=tree_numeric_cols,
    categorical_columns=pruned_tree_cat_cols,
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
    "Lasso": lasso_model,
    "Ridge": ridge_model,
    "GBT": gbt_model,
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

# MAGIC %md
# MAGIC ## Step 13 — Final Combined Results (All Models + Stacking)

# COMMAND ----------

all_test_results = pd.concat(
    [test_results, pd.DataFrame([stacking["stacked_test_metrics"]])],
    ignore_index=True,
    sort=False,
).sort_values(["RMSE", "model"]).reset_index(drop=True)

print("=== Final Test Set Results ===\n")
all_test_results

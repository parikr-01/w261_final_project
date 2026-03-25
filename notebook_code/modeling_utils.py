from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


DEFAULT_DELAY_THRESHOLD = 15.0
DEFAULT_SEVERE_DELAY_THRESHOLD = 60.0

RAW_NUMERIC_SPECS: Dict[str, bool] = {
    "HourlyVisibility": False,
    "HourlyWindSpeed": False,
    "HourlyPressureChange": False,
    "HourlyRelativeHumidity": False,
    "HourlyPrecipitation": True,
    "DailyPrecipitation": True,
    "DailySnowDepth": True,
    "DailySnowfall": True,
}

ENGINEERED_NUMERIC_CANDIDATES = [
    "DISTANCE",
    "DAY_OF_WEEK",
    "MONTH",
    "QUARTER",
    "scheduled_dep_hour",
    "is_weekend",
]

WEATHER_FLAG_COLUMNS = [
    "has_rain",
    "has_snow",
    "has_fog",
    "has_thunder",
]

LINEAR_CATEGORICAL_CANDIDATES = [
    "OP_CARRIER",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "DEP_TIME_BLK",
    "ORIGIN_STATE_ABR",
    "DEST_STATE_ABR",
]

GBT_CATEGORICAL_CANDIDATES = [
    "OP_CARRIER",
    "OP_UNIQUE_CARRIER",
    "DEP_TIME_BLK",
]


def _unique(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator in (0, None):
        return None
    return float(numerator) / float(denominator)


def _first_available(df: DataFrame, candidates: Sequence[str]) -> List[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return [candidate]
    return []


def _numeric_from_string(column_name: str, trace_to_zero: bool = False):
    raw = F.trim(F.col(column_name).cast("string"))
    numeric_text = F.regexp_extract(raw, r"-?\d+(?:\.\d+)?", 0)
    trace_marker = raw.rlike(r"^[Tt](?:s)?$")

    return (
        F.when(raw.isNull() | (raw == "") | raw.isin("NULL", "null", "NA", "na", "M", "m", "*"), None)
        .when(F.lit(trace_to_zero) & trace_marker, F.lit(0.0))
        .when(numeric_text == "", None)
        .otherwise(numeric_text.cast("double"))
    )


def prepare_modeling_frame(
    df: DataFrame,
    target_col: str = "DEP_DELAY",
    date_col: str = "FL_DATE",
    sample_fraction: Optional[float] = None,
    seed: int = 42,
) -> DataFrame:
    missing = [column for column in [target_col, date_col] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = df
    if sample_fraction is not None:
        frame = frame.sample(withReplacement=False, fraction=sample_fraction, seed=seed)

    frame = frame.withColumn(date_col, F.to_date(F.col(date_col)))
    frame = frame.filter(F.col(date_col).isNotNull()).filter(F.col(target_col).isNotNull())

    if "CANCELLED" in frame.columns:
        frame = frame.filter(F.coalesce(F.col("CANCELLED").cast("double"), F.lit(0.0)) == 0.0)
    if "DIVERTED" in frame.columns:
        frame = frame.filter(F.coalesce(F.col("DIVERTED").cast("double"), F.lit(0.0)) == 0.0)

    if "DISTANCE" in frame.columns:
        frame = frame.withColumn("DISTANCE", F.col("DISTANCE").cast("double"))
    for base_column in ["DAY_OF_WEEK", "MONTH", "QUARTER"]:
        if base_column in frame.columns:
            frame = frame.withColumn(base_column, F.col(base_column).cast("double"))

    if "CRS_DEP_TIME" in frame.columns:
        dep_time_text = F.lpad(
            F.coalesce(F.col("CRS_DEP_TIME").cast("int").cast("string"), F.lit("0")),
            4,
            "0",
        )
        frame = frame.withColumn("scheduled_dep_hour", F.substring(dep_time_text, 1, 2).cast("double"))
    else:
        frame = frame.withColumn("scheduled_dep_hour", F.lit(None).cast("double"))

    if "DAY_OF_WEEK" in frame.columns:
        frame = frame.withColumn(
            "is_weekend",
            F.when(F.col("DAY_OF_WEEK").isin(6.0, 7.0), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    else:
        frame = frame.withColumn("is_weekend", F.lit(0.0))

    if "HourlyPresentWeatherType" in frame.columns:
        weather_text = F.upper(F.coalesce(F.col("HourlyPresentWeatherType").cast("string"), F.lit("")))
        frame = frame.withColumn("has_rain", F.when(weather_text.rlike(r"RA|DZ|SH"), 1.0).otherwise(0.0))
        frame = frame.withColumn("has_snow", F.when(weather_text.rlike(r"SN|SG|PL|IC"), 1.0).otherwise(0.0))
        frame = frame.withColumn("has_fog", F.when(weather_text.rlike(r"FG|BR|HZ"), 1.0).otherwise(0.0))
        frame = frame.withColumn("has_thunder", F.when(weather_text.rlike(r"TS"), 1.0).otherwise(0.0))
    else:
        for flag_column in WEATHER_FLAG_COLUMNS:
            frame = frame.withColumn(flag_column, F.lit(0.0))

    parsed_numeric_columns: List[str] = []
    for raw_column, trace_to_zero in RAW_NUMERIC_SPECS.items():
        parsed_column = f"{raw_column}_num"
        if raw_column in frame.columns:
            frame = frame.withColumn(parsed_column, _numeric_from_string(raw_column, trace_to_zero=trace_to_zero))
            parsed_numeric_columns.append(parsed_column)

    frame = frame.withColumn("label", F.col(target_col).cast("double"))

    keep_columns = _unique(
        [date_col, "label"]
        + [column for column in ENGINEERED_NUMERIC_CANDIDATES if column in frame.columns]
        + parsed_numeric_columns
        + [column for column in WEATHER_FLAG_COLUMNS if column in frame.columns]
        + [column for column in LINEAR_CATEGORICAL_CANDIDATES if column in frame.columns]
    )

    return frame.select(*keep_columns)


def infer_time_split_boundaries(
    df: DataFrame,
    date_col: str = "FL_DATE",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> Tuple[str, str]:
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must be positive and sum to less than 1.")

    dates = [
        row[0]
        for row in df.select(date_col)
        .where(F.col(date_col).isNotNull())
        .distinct()
        .orderBy(date_col)
        .collect()
    ]

    if len(dates) < 3:
        raise ValueError("Need at least three distinct dates to infer train/validation/test boundaries.")

    train_index = max(0, min(len(dates) - 3, int(len(dates) * train_ratio) - 1))
    valid_index = max(train_index + 1, min(len(dates) - 2, int(len(dates) * (train_ratio + valid_ratio)) - 1))

    return dates[train_index].isoformat(), dates[valid_index].isoformat()


def time_based_split(
    df: DataFrame,
    date_col: str,
    train_end: str,
    valid_end: str,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    train_end_date = F.to_date(F.lit(str(train_end)))
    valid_end_date = F.to_date(F.lit(str(valid_end)))

    train_df = df.filter(F.col(date_col) <= train_end_date)
    valid_df = df.filter((F.col(date_col) > train_end_date) & (F.col(date_col) <= valid_end_date))
    test_df = df.filter(F.col(date_col) > valid_end_date)

    return train_df, valid_df, test_df


def describe_splits(
    train_df: DataFrame,
    valid_df: DataFrame,
    test_df: DataFrame,
    date_col: str = "FL_DATE",
) -> pd.DataFrame:
    rows = []
    for split_name, frame in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
        stats = (
            frame.agg(
                F.count("*").alias("rows"),
                F.min(F.col(date_col)).alias("min_date"),
                F.max(F.col(date_col)).alias("max_date"),
            )
            .collect()[0]
            .asDict()
        )
        stats["split"] = split_name
        rows.append(stats)

    ordered = pd.DataFrame(rows)
    return ordered[["split", "rows", "min_date", "max_date"]]


def resolve_feature_columns(df: DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_columns = _unique(
        [column for column in ENGINEERED_NUMERIC_CANDIDATES if column in df.columns]
        + [f"{column}_num" for column in RAW_NUMERIC_SPECS if f"{column}_num" in df.columns]
        + [column for column in WEATHER_FLAG_COLUMNS if column in df.columns]
    )
    linear_categorical = _unique(
        _first_available(df, ["OP_CARRIER", "OP_UNIQUE_CARRIER"])
        + [column for column in ["ORIGIN", "DEST", "DEP_TIME_BLK", "ORIGIN_STATE_ABR", "DEST_STATE_ABR"] if column in df.columns]
    )
    gbt_categorical = _unique(
        _first_available(df, ["OP_CARRIER", "OP_UNIQUE_CARRIER"])
        + [column for column in ["DEP_TIME_BLK"] if column in df.columns]
    )

    return numeric_columns, linear_categorical, gbt_categorical


def prune_empty_feature_columns(
    df: DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Tuple[List[str], List[str]]:
    candidate_columns = _unique(list(numeric_columns) + list(categorical_columns))
    if not candidate_columns:
        raise ValueError("No feature columns were provided.")

    counts = (
        df.select([F.count(F.col(column)).alias(column) for column in candidate_columns])
        .collect()[0]
        .asDict()
    )

    usable_numeric = [column for column in numeric_columns if counts.get(column, 0) > 0]
    usable_categorical = [column for column in categorical_columns if counts.get(column, 0) > 0]

    if not usable_numeric and not usable_categorical:
        raise ValueError("All candidate feature columns are empty after filtering.")

    return usable_numeric, usable_categorical


def _build_linear_pipeline(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    reg_param: float,
    elastic_net_param: float,
    max_iter: int = 100,
) -> Pipeline:
    stages = []
    feature_inputs: List[str] = []

    if categorical_columns:
        indexed_columns = [f"{column}_idx" for column in categorical_columns]
        encoded_columns = [f"{column}_ohe" for column in categorical_columns]

        stages.extend(
            [
                StringIndexer(inputCol=column, outputCol=indexed, handleInvalid="keep")
                for column, indexed in zip(categorical_columns, indexed_columns)
            ]
        )
        stages.append(
            OneHotEncoder(inputCols=indexed_columns, outputCols=encoded_columns, handleInvalid="keep")
        )
        feature_inputs.extend(encoded_columns)

    if numeric_columns:
        imputed_columns = [f"{column}_imp" for column in numeric_columns]
        stages.append(
            Imputer(strategy="median", inputCols=list(numeric_columns), outputCols=imputed_columns)
        )
        stages.append(
            VectorAssembler(
                inputCols=imputed_columns,
                outputCol="numeric_features",
                handleInvalid="keep",
            )
        )
        stages.append(
            StandardScaler(
                inputCol="numeric_features",
                outputCol="scaled_numeric_features",
                withStd=True,
                withMean=False,
            )
        )
        feature_inputs.insert(0, "scaled_numeric_features")

    stages.append(
        VectorAssembler(inputCols=feature_inputs, outputCol="features", handleInvalid="keep")
    )
    stages.append(
        LinearRegression(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            regParam=float(reg_param),
            elasticNetParam=float(elastic_net_param),
            standardization=False,
            maxIter=int(max_iter),
        )
    )

    return Pipeline(stages=stages)


def _build_gbt_pipeline(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    max_depth: int,
    max_iter: int,
    step_size: float,
    seed: int = 42,
) -> Pipeline:
    stages = []
    feature_inputs: List[str] = []

    if categorical_columns:
        indexed_columns = [f"{column}_idx" for column in categorical_columns]
        encoded_columns = [f"{column}_ohe" for column in categorical_columns]

        stages.extend(
            [
                StringIndexer(inputCol=column, outputCol=indexed, handleInvalid="keep")
                for column, indexed in zip(categorical_columns, indexed_columns)
            ]
        )
        stages.append(
            OneHotEncoder(inputCols=indexed_columns, outputCols=encoded_columns, handleInvalid="keep")
        )
        feature_inputs.extend(encoded_columns)

    if numeric_columns:
        imputed_columns = [f"{column}_imp" for column in numeric_columns]
        stages.append(
            Imputer(strategy="median", inputCols=list(numeric_columns), outputCols=imputed_columns)
        )
        feature_inputs = imputed_columns + feature_inputs

    stages.append(
        VectorAssembler(inputCols=feature_inputs, outputCol="features", handleInvalid="keep")
    )
    stages.append(
        GBTRegressor(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            maxDepth=int(max_depth),
            maxIter=int(max_iter),
            stepSize=float(step_size),
            subsamplingRate=0.8,
            seed=int(seed),
        )
    )

    return Pipeline(stages=stages)


def evaluate_predictions(
    predictions: DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
    delay_threshold: float = DEFAULT_DELAY_THRESHOLD,
    severe_delay_threshold: float = DEFAULT_SEVERE_DELAY_THRESHOLD,
) -> Dict[str, Optional[float]]:
    clean_predictions = predictions.where(
        F.col(label_col).isNotNull() & F.col(prediction_col).isNotNull()
    )

    evaluators = {
        "RMSE": RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName="rmse",
        ),
        "MAE": RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName="mae",
        ),
        "R2": RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName="r2",
        ),
    }

    metrics = {name: evaluator.evaluate(clean_predictions) for name, evaluator in evaluators.items()}

    scored = (
        clean_predictions.select(label_col, prediction_col)
        .withColumn("actual_delayed", (F.col(label_col) > F.lit(delay_threshold)).cast("int"))
        .withColumn("predicted_delayed", (F.col(prediction_col) > F.lit(delay_threshold)).cast("int"))
        .withColumn("actual_severe", (F.col(label_col) > F.lit(severe_delay_threshold)).cast("int"))
        .withColumn("predicted_severe", (F.col(prediction_col) > F.lit(severe_delay_threshold)).cast("int"))
    )

    counts = (
        scored.agg(
            F.count("*").alias("rows"),
            F.sum(
                F.when((F.col("actual_delayed") == 1) & (F.col("predicted_delayed") == 1), 1).otherwise(0)
            ).alias("tp_delayed"),
            F.sum(
                F.when((F.col("actual_delayed") == 0) & (F.col("predicted_delayed") == 0), 1).otherwise(0)
            ).alias("tn_delayed"),
            F.sum(
                F.when((F.col("actual_delayed") == 0) & (F.col("predicted_delayed") == 1), 1).otherwise(0)
            ).alias("fp_delayed"),
            F.sum(
                F.when((F.col("actual_delayed") == 1) & (F.col("predicted_delayed") == 0), 1).otherwise(0)
            ).alias("fn_delayed"),
            F.sum(
                F.when((F.col("actual_severe") == 1) & (F.col("predicted_severe") == 1), 1).otherwise(0)
            ).alias("tp_severe"),
            F.sum(
                F.when((F.col("actual_severe") == 1) & (F.col("predicted_severe") == 0), 1).otherwise(0)
            ).alias("fn_severe"),
        )
        .collect()[0]
        .asDict()
    )

    precision_delayed = _safe_divide(
        counts["tp_delayed"],
        counts["tp_delayed"] + counts["fp_delayed"],
    )
    recall_delayed = _safe_divide(
        counts["tp_delayed"],
        counts["tp_delayed"] + counts["fn_delayed"],
    )

    metrics["OTPA"] = _safe_divide(
        counts["tp_delayed"] + counts["tn_delayed"],
        counts["rows"],
    )
    metrics["SDDR"] = _safe_divide(
        counts["tp_severe"],
        counts["tp_severe"] + counts["fn_severe"],
    )
    metrics["F1"] = (
        None
        if precision_delayed is None or recall_delayed is None or (precision_delayed + recall_delayed) == 0
        else 2 * precision_delayed * recall_delayed / (precision_delayed + recall_delayed)
    )
    metrics["rows"] = counts["rows"]

    return metrics


def search_linear_model(
    train_df: DataFrame,
    valid_df: DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    reg_params: Sequence[float],
    elastic_net_param: float,
    model_name: str,
    max_iter: int = 100,
    delay_threshold: float = DEFAULT_DELAY_THRESHOLD,
    severe_delay_threshold: float = DEFAULT_SEVERE_DELAY_THRESHOLD,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    results = []
    best_row = None

    for reg_param in reg_params:
        pipeline = _build_linear_pipeline(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            reg_param=reg_param,
            elastic_net_param=elastic_net_param,
            max_iter=max_iter,
        )
        model = pipeline.fit(train_df)
        predictions = model.transform(valid_df).select("label", "prediction").cache()
        metrics = evaluate_predictions(
            predictions,
            delay_threshold=delay_threshold,
            severe_delay_threshold=severe_delay_threshold,
        )
        predictions.unpersist()

        row = {
            "model": model_name,
            "split": "validation",
            "regParam": float(reg_param),
            "elasticNetParam": float(elastic_net_param),
            "maxIter": int(max_iter),
            **metrics,
        }
        results.append(row)

        if best_row is None or row["RMSE"] < best_row["RMSE"]:
            best_row = row

    return best_row, pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)


def search_gbt_model(
    train_df: DataFrame,
    valid_df: DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    param_grid: Sequence[Dict[str, float]],
    model_name: str = "GBT",
    delay_threshold: float = DEFAULT_DELAY_THRESHOLD,
    severe_delay_threshold: float = DEFAULT_SEVERE_DELAY_THRESHOLD,
    seed: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    results = []
    best_row = None

    for params in param_grid:
        pipeline = _build_gbt_pipeline(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            max_depth=int(params["maxDepth"]),
            max_iter=int(params["maxIter"]),
            step_size=float(params["stepSize"]),
            seed=seed,
        )
        model = pipeline.fit(train_df)
        predictions = model.transform(valid_df).select("label", "prediction").cache()
        metrics = evaluate_predictions(
            predictions,
            delay_threshold=delay_threshold,
            severe_delay_threshold=severe_delay_threshold,
        )
        predictions.unpersist()

        row = {
            "model": model_name,
            "split": "validation",
            "maxDepth": int(params["maxDepth"]),
            "maxIter": int(params["maxIter"]),
            "stepSize": float(params["stepSize"]),
            **metrics,
        }
        results.append(row)

        if best_row is None or row["RMSE"] < best_row["RMSE"]:
            best_row = row

    return best_row, pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)


def fit_linear_model(
    train_df: DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    reg_param: float,
    elastic_net_param: float,
    max_iter: int = 100,
) -> PipelineModel:
    pipeline = _build_linear_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        reg_param=reg_param,
        elastic_net_param=elastic_net_param,
        max_iter=max_iter,
    )
    return pipeline.fit(train_df)


def fit_gbt_model(
    train_df: DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    max_depth: int,
    max_iter: int,
    step_size: float,
    seed: int = 42,
) -> PipelineModel:
    pipeline = _build_gbt_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        max_depth=max_depth,
        max_iter=max_iter,
        step_size=step_size,
        seed=seed,
    )
    return pipeline.fit(train_df)


def evaluate_model(
    model: PipelineModel,
    df: DataFrame,
    model_name: str,
    split_name: str,
    delay_threshold: float = DEFAULT_DELAY_THRESHOLD,
    severe_delay_threshold: float = DEFAULT_SEVERE_DELAY_THRESHOLD,
) -> Dict[str, Optional[float]]:
    predictions = model.transform(df).select("label", "prediction").cache()
    metrics = evaluate_predictions(
        predictions,
        delay_threshold=delay_threshold,
        severe_delay_threshold=severe_delay_threshold,
    )
    predictions.unpersist()

    return {
        "model": model_name,
        "split": split_name,
        **metrics,
    }


def _feature_metadata(model: PipelineModel, reference_df: DataFrame) -> List[str]:
    transformed = model.transform(reference_df.limit(1))
    metadata = transformed.schema["features"].metadata.get("ml_attr", {})
    attrs = metadata.get("attrs", {})
    flattened = []
    for key in ["numeric", "binary", "nominal"]:
        flattened.extend(attrs.get(key, []))

    if not flattened:
        if hasattr(model.stages[-1], "coefficients"):
            feature_count = len(model.stages[-1].coefficients)
        else:
            feature_count = model.stages[-1].featureImportances.size
        return [f"feature_{index}" for index in range(feature_count)]

    flattened = sorted(flattened, key=lambda item: item["idx"])
    return [item["name"] for item in flattened]


def top_linear_coefficients(
    model: PipelineModel,
    reference_df: DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    feature_names = _feature_metadata(model, reference_df)
    coefficients = list(model.stages[-1].coefficients)
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)


def top_gbt_importances(
    model: PipelineModel,
    reference_df: DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    feature_names = _feature_metadata(model, reference_df)
    importances = list(model.stages[-1].featureImportances.toArray())
    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    return table.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

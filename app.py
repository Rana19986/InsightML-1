# ============================================================
# InsightML — ADAPTIVE MACHINE LEARNING & DATA ANALYTICS
# Version 2.0 | Google Colab | Free
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report,
    confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    IsolationForest
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------
# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

# Uses the folder where app.py lives instead of Google Colab's /content
BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 200_000
MAX_COLUMNS = 100

print("InsightML v2.0 loaded.")



# ------------------------------------------------------------
# 2. COMMON HELPERS
# ------------------------------------------------------------

def read_csv_file(file_path):
    if file_path is None:
        raise gr.Error("Please upload a CSV file.")

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as exc:
        raise gr.Error(f"Could not read CSV: {exc}")

    if df.empty:
        raise gr.Error("The CSV file is empty.")

    if len(df) > MAX_ROWS:
        raise gr.Error(f"Maximum supported rows: {MAX_ROWS:,}.")

    if len(df.columns) > MAX_COLUMNS:
        raise gr.Error(f"Maximum supported columns: {MAX_COLUMNS}.")

    if not df.columns.is_unique:
        raise gr.Error("Duplicate column names detected. Rename them first.")

    return df


def empty_plot(message):
    fig = px.scatter(title=message)
    return fig


def get_column_choices(file_path):
    if file_path is None:
        return gr.update(choices=[], value=None)

    try:
        df = pd.read_csv(file_path, nrows=5)
        return gr.update(
            choices=df.columns.tolist(),
            value=None
        )
    except Exception:
        return gr.update(choices=[], value=None)


def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = [
        c for c in X.columns
        if c not in numeric_features
    ]

    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipe, numeric_features)
        )

    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipe, categorical_features)
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers)


def make_model(task, model_name):
    if task == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            )
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ),
            "Decision Tree": DecisionTreeRegressor(
                random_state=42
            )
        }

    if model_name not in models:
        raise ValueError("Unsupported model selected.")

    return models[model_name]


# ------------------------------------------------------------
# 3. DATA ANALYSIS ENGINE
# ------------------------------------------------------------

def analyze_dataset(file_path, target_column):

    df = read_csv_file(file_path)

    original_rows, original_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    datetime_columns = []

    for col in df.columns:
        if any(word in str(col).lower() for word in [
            "date", "datetime", "timestamp"
        ]):
            try:
                parsed = pd.to_datetime(
                    df[col], errors="coerce"
                )
                if parsed.notna().mean() >= 0.60:
                    df[col] = parsed
                    datetime_columns.append(col)
            except Exception:
                pass

    numeric_columns = [
        c for c in numeric_columns
        if c not in datetime_columns
    ]

    categorical_columns = [
        c for c in df.columns
        if c not in numeric_columns
        and c not in datetime_columns
    ]

    quality_df = pd.DataFrame([
        {
            "Column": str(c),
            "Data Type": str(df[c].dtype),
            "Missing Values": int(df[c].isna().sum()),
            "Missing %": round(df[c].isna().mean() * 100, 2),
            "Unique Values": int(df[c].nunique(dropna=True))
        }
        for c in df.columns
    ])

    summary = f"""
# InsightML — Dataset Analysis

## Dataset overview

- **Rows:** {original_rows:,}
- **Columns:** {original_cols:,}
- **Duplicate rows:** {duplicate_rows:,}
- **Missing cells:** {missing_cells:,}
- **Numeric columns:** {len(numeric_columns)}
- **Categorical columns:** {len(categorical_columns)}
- **Detected date columns:** {len(datetime_columns)}

## Available capabilities

- Automated data quality analysis
- Numeric statistics and distributions
- Correlation analysis
- Categorical analysis
- Date-based record analysis
- Conditional business KPIs
- ML classification and regression
- Model comparison and prediction
- Anomaly detection
"""

    if target_column and target_column in df.columns:
        summary += f"\n\n**Selected ML target:** `{target_column}`"

    # Numeric summary
    if numeric_columns:
        numeric_summary = (
            df[numeric_columns]
            .describe()
            .T.reset_index()
            .rename(columns={"index": "Column"})
            .round(3)
        )

        selected_num = numeric_columns[0]
        distribution_chart = px.histogram(
            df,
            x=selected_num,
            title=f"Distribution: {selected_num}",
            marginal="box"
        )
    else:
        numeric_summary = pd.DataFrame({
            "Message": ["No numeric columns found."]
        })
        distribution_chart = empty_plot(
            "No numeric columns available"
        )

    # Correlation
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr()
        correlation_chart = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Numeric Correlation Heatmap",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1
        )
        correlation_df = (
            corr.round(3).reset_index()
            .rename(columns={"index": "Column"})
        )
    else:
        correlation_chart = empty_plot(
            "At least two numeric columns required"
        )
        correlation_df = pd.DataFrame({
            "Message": ["Not enough numeric columns."]
        })

    # Categorical analysis
    if categorical_columns:
        cat_col = categorical_columns[0]
        category_summary = (
            df[cat_col].fillna("(Missing)").astype(str)
            .value_counts().head(15)
            .rename_axis("Category")
            .reset_index(name="Count")
        )
        category_chart = px.bar(
            category_summary,
            x="Category",
            y="Count",
            title=f"Top categories: {cat_col}"
        )
    else:
        category_summary = pd.DataFrame({
            "Message": ["No categorical columns found."]
        })
        category_chart = empty_plot(
            "No categorical columns available"
        )

    # Date analysis
    if datetime_columns:
        date_col = datetime_columns[0]
        date_df = pd.DataFrame({
            "Date": df[date_col].dt.date
        }).dropna()

        date_summary = (
            date_df.groupby("Date").size()
            .reset_index(name="Records")
        )

        date_chart = px.line(
            date_summary,
            x="Date",
            y="Records",
            title=f"Records over time: {date_col}"
        )
    else:
        date_summary = pd.DataFrame({
            "Message": ["No suitable date column detected."]
        })
        date_chart = empty_plot("No date column detected")

    # Business KPIs
    lower_map = {
        str(c).lower().strip(): c for c in df.columns
    }

    revenue_names = [
        "revenue", "sales", "amount", "total_amount",
        "total_sales", "total_revenue"
    ]
    quantity_names = [
        "quantity", "units_sold", "units", "qty"
    ]

    revenue_col = next(
        (lower_map[n] for n in revenue_names if n in lower_map),
        None
    )
    quantity_col = next(
        (lower_map[n] for n in quantity_names if n in lower_map),
        None
    )

    if revenue_col:
        values = pd.to_numeric(
            df[revenue_col], errors="coerce"
        )

        kpis = [
            {
                "Metric": f"Total {revenue_col}",
                "Value": round(float(values.sum()), 2)
            },
            {
                "Metric": f"Average {revenue_col}",
                "Value": round(float(values.mean()), 2)
            }
        ]

        if quantity_col:
            qty = pd.to_numeric(
                df[quantity_col], errors="coerce"
            )
            kpis.append({
                "Metric": f"Total {quantity_col}",
                "Value": round(float(qty.sum()), 2)
            })

        kpi_df = pd.DataFrame(kpis)
    else:
        kpi_df = pd.DataFrame({
            "Message": [
                "No recognized revenue or sales column detected."
            ]
        })

    # Reports
    quality_path = REPORT_DIR / "data_quality.csv"
    numeric_path = REPORT_DIR / "numeric_summary.csv"
    category_path = REPORT_DIR / "category_summary.csv"
    kpi_path = REPORT_DIR / "business_kpis.csv"
    report_path = REPORT_DIR / "analysis_report.txt"

    quality_df.to_csv(quality_path, index=False)
    numeric_summary.to_csv(numeric_path, index=False)
    category_summary.to_csv(category_path, index=False)
    kpi_df.to_csv(kpi_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n\nDATA QUALITY\n")
        f.write(quality_df.to_string(index=False))
        f.write("\n\nNUMERIC SUMMARY\n")
        f.write(numeric_summary.to_string(index=False))
        f.write("\n\nBUSINESS KPIs\n")
        f.write(kpi_df.to_string(index=False))

    return (
        summary, quality_df, numeric_summary,
        distribution_chart, correlation_chart,
        correlation_df, category_chart,
        category_summary, date_chart, date_summary,
        kpi_df, str(report_path)
    )


# ------------------------------------------------------------
# 4. MACHINE LEARNING TRAINING
# ------------------------------------------------------------

def train_ml_model(file_path, target_column, task, model_name,
                   test_size, remove_id_columns):

    df = read_csv_file(file_path)

    if not target_column or target_column not in df.columns:
        raise gr.Error("Select a valid target column.")

    if df[target_column].isna().all():
        raise gr.Error("The target column contains only missing values.")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column]).copy()

    if len(df) < 10:
        raise gr.Error("At least 10 usable rows are recommended.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Remove date columns and optionally identifier-like columns
    date_cols = []
    for c in X.columns:
        if any(w in str(c).lower() for w in [
            "date", "datetime", "timestamp"
        ]):
            parsed = pd.to_datetime(X[c], errors="coerce")
            if parsed.notna().mean() >= 0.60:
                date_cols.append(c)

    if date_cols:
        X = X.drop(columns=date_cols)

    if remove_id_columns:
        id_cols = [
            c for c in X.columns
            if str(c).lower().strip() in [
                "id", "customer_id", "user_id",
                "row_id", "index", "record_id"
            ]
        ]
        if id_cols:
            X = X.drop(columns=id_cols)

    # Remove columns that contain no usable information
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] == 0:
        raise gr.Error("No usable feature columns remain.")

    # Determine the task and validate target
    if task == "Classification":
        y = y.astype(str)

        if y.nunique() < 2:
            raise gr.Error(
                "Classification requires at least two target classes."
            )

        if y.nunique() > 100:
            raise gr.Error(
                "Too many target classes. Select a more suitable target."
            )

        stratify = y if y.value_counts().min() >= 2 else None

    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        if len(y) < 10:
            raise gr.Error(
                "Not enough valid numeric target values for regression."
            )

        stratify = None

    # Keep model size manageable on free Colab
    if len(X) > 50_000:
        sampled = X.sample(
            n=50_000, random_state=42
        ).index
        X = X.loc[sampled]
        y = y.loc[sampled]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(test_size),
            random_state=42,
            stratify=stratify
        )

        preprocessor = build_preprocessor(X_train)
        estimator = make_model(task, model_name)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    except Exception as exc:
        raise gr.Error(f"Training failed: {exc}")

    # Metrics
    if task == "Classification":
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )

        metrics = pd.DataFrame([
            {"Metric": "Accuracy", "Value": round(accuracy, 4)},
            {"Metric": "Weighted Precision", "Value": round(precision, 4)},
            {"Metric": "Weighted Recall", "Value": round(recall, 4)},
            {"Metric": "Weighted F1", "Value": round(f1, 4)}
        ])

        labels = sorted(pd.Series(y_test).astype(str).unique())
        cm = confusion_matrix(
            y_test.astype(str),
            pd.Series(predictions).astype(str),
            labels=labels
        )

        confusion_df = pd.DataFrame(
            cm, index=labels, columns=labels
        ).reset_index().rename(columns={"index": "Actual"})
        confusion_chart = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            title="Confusion Matrix",
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count"
            }
        )

        extra = (
            f"Accuracy: {accuracy:.4f}\n"
            f"Weighted F1: {f1:.4f}\n"
        )

    else:
        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)

        metrics = pd.DataFrame([
            {"Metric": "MAE", "Value": round(mae, 4)},
            {"Metric": "RMSE", "Value": round(rmse, 4)},
            {"Metric": "R²", "Value": round(r2, 4)}
        ])

        comparison_df = pd.DataFrame({
            "Actual": y_test.to_numpy(),
            "Predicted": predictions
        })

        confusion_chart = px.scatter(
            comparison_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted",
            trendline=None
        )
        confusion_df = comparison_df.head(100)

        extra = (
            f"MAE: {mae:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"R²: {r2:.4f}\n"
        )

    # Save the model and metadata
    model_path = MODEL_DIR / "latest_model.joblib"
    joblib.dump({
        "pipeline": model,
        "task": task,
        "target": target_column,
        "features": X.columns.tolist(),
        "model_name": model_name
    }, model_path)

    # Save test predictions
    prediction_path = REPORT_DIR / "test_predictions.csv"
    pd.DataFrame({
        "Actual": np.asarray(y_test),
        "Predicted": np.asarray(predictions)
    }).to_csv(prediction_path, index=False)

    report_path = REPORT_DIR / "ml_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("InsightML — ML Training Report\n")
        f.write(f"Task: {task}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Target: {target_column}\n")
        f.write(f"Training rows: {len(X_train)}\n")
        f.write(f"Testing rows: {len(X_test)}\n\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nNote: Test metrics are estimates for this split.")

    status = f"""
## Model trained successfully

- **Task:** {task}
- **Model:** {model_name}
- **Target:** {target_column}
- **Training rows:** {len(X_train):,}
- **Testing rows:** {len(X_test):,}
- **Features used:** {X.shape[1]}

### Evaluation

{extra}

The evaluation uses a held-out test set. Results may vary with other
data splits and do not guarantee future performance.
"""

    return (
        status, metrics, confusion_chart,
        confusion_df, str(model_path), str(report_path)
    )


# ------------------------------------------------------------
# 5.MODEL COMPARISON
# ------------------------------------------------------------

def compare_models(file_path, target_column, task, test_size):

    df = read_csv_file(file_path)

    if not target_column or target_column not in df.columns:
        raise gr.Error("Select a valid target column.")

    df = df.dropna(subset=[target_column]).copy()

    y = df[target_column]
    X = df.drop(columns=[target_column])

    date_cols = []
    for c in X.columns:
        if any(w in str(c).lower() for w in [
            "date", "datetime", "timestamp"
        ]):
            parsed = pd.to_datetime(X[c], errors="coerce")
            if parsed.notna().mean() >= 0.60:
                date_cols.append(c)

    if date_cols:
        X = X.drop(columns=date_cols)

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.empty:
        raise gr.Error("No usable features found.")

    if task == "Classification":
        y = y.astype(str)
        if y.nunique() < 2:
            raise gr.Error("Classification requires at least two classes.")
        stratify = y if y.value_counts().min() >= 2 else None
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        stratify = None

    if len(X) > 30_000:
        sampled = X.sample(30_000, random_state=42).index
        X, y = X.loc[sampled], y.loc[sampled]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(test_size),
            random_state=42,
            stratify=stratify
        )
    except Exception as exc:
        raise gr.Error(f"Could not create train/test split: {exc}")

    names = (
        ["Logistic Regression", "Decision Tree", "Random Forest"]
        if task == "Classification"
        else ["Linear Regression", "Decision Tree", "Random Forest"]
    )

    results = []

    for name in names:
        try:
            pipe = Pipeline([
                ("preprocessor", build_preprocessor(X_train)),
                ("model", make_model(task, name))
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            if task == "Classification":
                results.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, pred), 4),
                    "Weighted F1": round(
                        f1_score(
                            y_test, pred,
                            average="weighted",
                            zero_division=0
                        ), 4
                    )
                })
            else:
                results.append({
                    "Model": name,
                    "MAE": round(
                        mean_absolute_error(y_test, pred), 4
                    ),
                    "RMSE": round(
                        float(np.sqrt(
                            mean_squared_error(y_test, pred)
                        )), 4
                    ),
                    "R²": round(r2_score(y_test, pred), 4)
                })

        except Exception as exc:
            results.append({
                "Model": name,
                "Error": str(exc)[:200]
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )

    return result_df


# ------------------------------------------------------------
# 6. ANOMALY DETECTION
# ------------------------------------------------------------

def detect_anomalies(file_path, contamination):

    df = read_csv_file(file_path)

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.shape[1] == 0:
        raise gr.Error("Anomaly detection requires numeric columns.")

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf], np.nan
    )

    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        raise gr.Error("No usable numeric data found.")

    if len(numeric_df) < 10:
        raise gr.Error("At least 10 rows are recommended.")

    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )

    if len(numeric_df) > 50_000:
        sample_idx = numeric_df.sample(
            50_000, random_state=42
        ).index
        data = numeric_df.loc[sample_idx]
        original_indices = sample_idx
    else:
        data = numeric_df
        original_indices = data.index

    detector = IsolationForest(
        contamination=float(contamination),
        random_state=42,
        n_estimators=100
    )

    labels = detector.fit_predict(data)

    result = df.loc[original_indices].copy()
    result["Anomaly"] = np.where(
        labels == -1, "Potential anomaly", "Normal"
    )
    result["Anomaly Score"] = detector.decision_function(data)

    result_path = REPORT_DIR / "anomaly_detection.csv"
    result.to_csv(result_path, index=False)

    counts = result["Anomaly"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]

    chart = px.bar(
        counts,
        x="Status",
        y="Count",
        title="Anomaly Detection Results"
    )

    status = f"""
## Anomaly detection complete

- Rows examined: {len(result):,}
- Potential anomalies: {(labels == -1).sum():,}
- Contamination setting: {float(contamination):.3f}

Isolation Forest flags unusual patterns in the selected numeric
features. A flag is not proof of fraud, error, or misconduct.
"""

    return status, result.head(500), chart, str(result_path)


# ------------------------------------------------------------
# 7. PREDICT USING THE SAVED MODEL
# ------------------------------------------------------------

def predict_with_saved_model(file_path):

    if not file_path:
        raise gr.Error("Upload a CSV containing the model's feature columns.")

    model_path = MODEL_DIR / "latest_model.joblib"

    if not model_path.exists():
        raise gr.Error("Train an ML model first.")

    try:
        bundle = joblib.load(model_path)
        model = bundle["pipeline"]
        features = bundle["features"]
        task = bundle["task"]

        new_df = read_csv_file(file_path)

        missing = [c for c in features if c not in new_df.columns]

        if missing:
            raise gr.Error(
                "Missing required feature columns: "
                + ", ".join(map(str, missing[:15]))
            )

        X_new = new_df[features].copy()
        predictions = model.predict(X_new)

        result = new_df.copy()
        result["Prediction"] = predictions

        if task == "Classification" and hasattr(
            model.named_steps["model"], "predict_proba"
        ):
            try:
                probabilities = model.predict_proba(X_new)
                result["Prediction Confidence"] = probabilities.max(axis=1)
            except Exception:
                pass

        prediction_path = REPORT_DIR / "new_predictions.csv"
        result.to_csv(prediction_path, index=False)

        status = f"""
## Prediction complete

- Task: {task}
- Model: {bundle['model_name']}
- Predicted rows: {len(result):,}

Predictions depend on the quality and distribution of the input data.
"""

        return status, result.head(500), str(prediction_path)

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Prediction failed: {exc}")


# ------------------------------------------------------------
# 8. USER INTERFACE
# ------------------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}
#main-title {
    text-align: center;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CSS,
    title="InsightML | Adaptive ML Analytics"
) as demo:

    gr.Markdown(
        """
        # 🔍 InsightML
        ### Adaptive Machine Learning & Business Analytics

        Upload a CSV dataset to explore data quality, statistics,
        business KPIs, ML training, model comparison, prediction,
        and anomaly detection.
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        """
        **Privacy note:** Uploaded data is processed by this running
        application. Do not upload sensitive personal or confidential
        business information to a public demo.
        """
    )

    with gr.Tab("Data Analysis"):

        analysis_file = gr.File(
            label="Upload CSV",
            file_types=[".csv"],
            type="filepath"
        )

        analysis_target = gr.Dropdown(
            label="Optional target column",
            choices=[],
            value=None
        )

        analysis_file.change(
            get_column_choices,
            inputs=analysis_file,
            outputs=analysis_target
        )

        analyze_button = gr.Button(
            "Analyze Dataset", variant="primary"
        )

        analysis_summary = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Data Quality"):
                quality_output = gr.Dataframe(
                    interactive=False
                )

            with gr.Tab("Numeric Analysis"):
                numeric_output = gr.Dataframe(
                    interactive=False
                )
                distribution_output = gr.Plot()

            with gr.Tab("Correlations"):
                correlation_plot = gr.Plot()
                correlation_table = gr.Dataframe(
                    interactive=False
                )

            with gr.Tab("Categories"):
                category_plot = gr.Plot()
                category_table = gr.Dataframe(
                    interactive=False
                )

            with gr.Tab("Time Analysis"):
                date_plot = gr.Plot()
                date_table = gr.Dataframe(
                    interactive=False
                )

            with gr.Tab("Business KPIs"):
                kpi_output = gr.Dataframe(
                    interactive=False
                )

        analysis_report = gr.File(
            label="Download analysis report"
        )

        analyze_button.click(
            analyze_dataset,
            inputs=[analysis_file, analysis_target],
            outputs=[
                analysis_summary,
                quality_output,
                numeric_output,
                distribution_output,
                correlation_plot,
                correlation_table,
                category_plot,
                category_table,
                date_plot,
                date_table,
                kpi_output,
                analysis_report
            ]
        )

    with gr.Tab("ML Training"):

        ml_file = gr.File(
            label="Upload training CSV",
            file_types=[".csv"],
            type="filepath"
        )

        ml_target = gr.Dropdown(
            label="Target column",
            choices=[],
            value=None
        )

        ml_file.change(
            get_column_choices,
            inputs=ml_file,
            outputs=ml_target
        )

        ml_task = gr.Radio(
            choices=["Classification", "Regression"],
            value="Classification",
            label="Machine learning task"
        )

        ml_model = gr.Dropdown(
            choices=[
                "Logistic Regression",
                "Random Forest",
                "Decision Tree"
            ],
            value="Random Forest",
            label="Classification model"
        )

        def update_model_choices(task):
            if task == "Classification":
                return gr.update(
                    choices=[
                        "Logistic Regression",
                        "Random Forest",
                        "Decision Tree"
                    ],
                    value="Random Forest"
                )
            return gr.update(
                choices=[
                    "Linear Regression",
                    "Random Forest",
                    "Decision Tree"
                ],
                value="Random Forest"
            )

        ml_task.change(
            update_model_choices,
            inputs=ml_task,
            outputs=ml_model
        )

        ml_test_size = gr.Slider(
            minimum=0.1,
            maximum=0.4,
            value=0.2,
            step=0.05,
            label="Test set proportion"
        )

        ml_remove_ids = gr.Checkbox(
            value=True,
            label="Remove common identifier columns"
        )

        train_button = gr.Button(
            "Train ML Model", variant="primary"
        )

        ml_status = gr.Markdown()
        ml_metrics = gr.Dataframe(interactive=False)
        ml_chart = gr.Plot()
        ml_details = gr.Dataframe(interactive=False)

        saved_model_file = gr.File(
            label="Saved model"
        )
        ml_report_file = gr.File(
            label="ML training report"
        )

        train_button.click(
            train_ml_model,
            inputs=[
                ml_file,
                ml_target,
                ml_task,
                ml_model,
                ml_test_size,
                ml_remove_ids
            ],
            outputs=[
                ml_status,
                ml_metrics,
                ml_chart,
                ml_details,
                saved_model_file,
                ml_report_file
            ]
        )

    with gr.Tab("Model Comparison"):

        compare_file = gr.File(
            label="Upload CSV",
            file_types=[".csv"],
            type="filepath"
        )

        compare_target = gr.Dropdown(
            label="Target column",
            choices=[],
            value=None
        )

        compare_file.change(
            get_column_choices,
            inputs=compare_file,
            outputs=compare_target
        )

        compare_task = gr.Radio(
            choices=["Classification", "Regression"],
            value="Classification",
            label="Task"
        )

        compare_test_size = gr.Slider(
            minimum=0.1,
            maximum=0.4,
            value=0.2,
            step=0.05,
            label="Test set proportion"
        )

        compare_button = gr.Button(
            "Compare Models", variant="primary"
        )

        compare_output = gr.Dataframe(
            interactive=False
        )

        compare_button.click(
            compare_models,
            inputs=[
                compare_file,
                compare_target,
                compare_task,
                compare_test_size
            ],
            outputs=compare_output
        )

    with gr.Tab("Prediction"):

        prediction_file = gr.File(
            label="Upload new CSV with feature columns",
            file_types=[".csv"],
            type="filepath"
        )

        predict_button = gr.Button(
            "Generate Predictions", variant="primary"
        )

        prediction_status = gr.Markdown()
        prediction_table = gr.Dataframe(
            interactive=False
        )
        prediction_download = gr.File(
            label="Download predictions"
        )

        predict_button.click(
            predict_with_saved_model,
            inputs=prediction_file,
            outputs=[
                prediction_status,
                prediction_table,
                prediction_download
            ]
        )

    with gr.Tab("Anomaly Detection"):

        anomaly_file = gr.File(
            label="Upload CSV",
            file_types=[".csv"],
            type="filepath"
        )

        contamination = gr.Slider(
            minimum=0.01,
            maximum=0.20,
            value=0.05,
            step=0.01,
            label="Expected anomaly proportion"
        )

        anomaly_button = gr.Button(
            "Detect Anomalies", variant="primary"
        )

        anomaly_status = gr.Markdown()
        anomaly_table = gr.Dataframe(
            interactive=False
        )
        anomaly_chart = gr.Plot()
        anomaly_download = gr.File(
            label="Download anomaly report"
        )

        anomaly_button.click(
            detect_anomalies,
            inputs=[anomaly_file, contamination],
            outputs=[
                anomaly_status,
                anomaly_table,
                anomaly_chart,
                anomaly_download
            ]
        )

    gr.Markdown(
        """
        ---
        **InsightML | Portfolio Project**

        Evaluation metrics are estimates, not guarantees of future
        performance. Anomaly flags require human review. Model
        predictions should not be used as the sole basis for
        high-impact decisions.
        """
    )

# ------------------------------------------------------------
# 9. LAUNCH
# ------------------------------------------------------------
print("Launching InsightML...")
print("Keep this cell running while using the public URL.")

demo.launch(
    share=True,
    debug=False,
    show_error=True
)
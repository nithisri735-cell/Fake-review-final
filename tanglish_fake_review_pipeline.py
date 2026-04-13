import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD


SLANG_MAP = {
    "semma": "excellent",
    "super": "excellent",
    "superr": "excellent",
    "mass": "great",
    "vera": "very",
    "vera level": "excellent",
    "mokka": "bad",
    "waste": "bad",
    "summa": "just",
    "seri": "okay",
    "sari": "okay",
    "romba": "very",
    "nalla": "good",
    "mosam": "bad",
    "kandippa": "definitely",
    "pidichiruku": "liked",
    "pidikkala": "disliked",
    "pannuthu": "does",
    "iruku": "is",
    "illa": "not",
    "ah": "",
    "la": "",
    "nu": "",
}

TANGLISH_TOKENS = {
    "ah",
    "ana",
    "apo",
    "appo",
    "iruku",
    "illa",
    "ipo",
    "kandippa",
    "la",
    "mokka",
    "nalla",
    "nu",
    "okva",
    "pannuthu",
    "pidichiruku",
    "romba",
    "semma",
    "seri",
    "summa",
    "vera",
}
LABEL_MAP = {
    "or": 0,
    "real": 0,
    "original": 0,
    "genuine": 0,
    "0": 0,
    0: 0,
    "cg": 1,
    "fake": 1,
    "spam": 1,
    "deceptive": 1,
    "1": 1,
    1: 1,
}




def parse_args():
    parser = argparse.ArgumentParser(
        description="Slang-aware fake review detection pipeline for English + Tanglish reviews"
    )
    parser.add_argument(
        "--english-dataset",
        default=r"C:\Users\SRINITHI R\Downloads\ML\fake reviews dataset.csv",
    )
    parser.add_argument(
        "--tanglish-dataset",
        default=r"C:\Users\SRINITHI R\Downloads\ML\tanglish_2000_reviews.csv",
    )
    parser.add_argument(
        "--extra-dataset",
        default="",
        help="Optional third Tanglish dataset path with review/text and label columns",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "artifacts"),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--run-transformer",
        action="store_true",
        help="Fine-tune a transformer model if torch and transformers are installed",
    )
    parser.add_argument(
        "--transformer-model",
        default="google/muril-base-cased",
        help="Recommended for code-mixed Indic text",
    )
    parser.add_argument(
        "--transformer-epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-text-features",
        type=int,
        default=25000,
    )
    parser.add_argument(
        "--near-duplicate-check",
        action="store_true",
        help="Run slower near-duplicate leakage detection in addition to exact overlap",
    )
    return parser.parse_args()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    lower_columns = {c.lower().strip(): c for c in df.columns}

    if "text_" in df.columns:
        rename_map["text_"] = "review"
    elif "text" in df.columns:
        rename_map["text"] = "review"
    elif "review_text" in df.columns:
        rename_map["review_text"] = "review"
    elif "review" not in df.columns:
        for candidate in ["content", "comment", "sentence"]:
            if candidate in lower_columns:
                rename_map[lower_columns[candidate]] = "review"
                break

    if "label" not in df.columns:
        for candidate in ["class", "target", "sentiment"]:
            if candidate in lower_columns:
                rename_map[lower_columns[candidate]] = "label"
                break

    return df.rename(columns=rename_map)


def normalize_label(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    return LABEL_MAP.get(value, np.nan)


def normalize_slang(text: str) -> str:
    text = str(text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = text.lower()
    for src, dst in sorted(SLANG_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    return text


def clean_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = normalize_slang(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_structured_features(review_series: pd.Series) -> pd.DataFrame:
    def one_row(text: str) -> dict:
        text = "" if pd.isna(text) else str(text)
        tokens = re.findall(r"\b\w+\b", text.lower())
        token_count = len(tokens)
        token_counter = Counter(tokens)
        slang_hits = sum(token_counter[token] for token in TANGLISH_TOKENS if token in token_counter)
        uppercase_chars = sum(1 for c in text if c.isupper())
        alpha_chars = sum(1 for c in text if c.isalpha())

        return {
            "char_count": len(text),
            "word_count": token_count,
            "avg_word_length": (sum(len(t) for t in tokens) / token_count) if token_count else 0.0,
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "digit_count": sum(c.isdigit() for c in text),
            "uppercase_ratio": (uppercase_chars / alpha_chars) if alpha_chars else 0.0,
            "slang_token_count": slang_hits,
            "slang_token_ratio": (slang_hits / token_count) if token_count else 0.0,
            "elongated_word_count": len(re.findall(r"\b\w*(\w)\1{2,}\w*\b", text.lower())),
        }

    return pd.DataFrame([one_row(text) for text in review_series])


class StructuredFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        else:
            series = pd.Series(X)
        features = build_structured_features(series)
        return features.values


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, score_func=chi2, k=15000):
        self.score_func = score_func
        self.k = k
        self.selector_ = None

    def fit(self, X, y):
        k = "all" if self.k == "all" else min(self.k, X.shape[1])
        self.selector_ = SelectKBest(score_func=self.score_func, k=k)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


def prepare_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = standardize_columns(df).copy()
    required_columns = {"review", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{source_name} is missing required columns: {sorted(missing_columns)}")

    if "category" not in df.columns:
        df["category"] = "unknown"
    if "rating" not in df.columns:
        df["rating"] = np.nan

    df["review"] = df["review"].astype(str).fillna("")
    df["label"] = df["label"].map(normalize_label)
    df["clean_review"] = df["review"].apply(clean_text)

    df = df.dropna(subset=["label"])
    df = df[df["clean_review"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_review"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["source"] = source_name
    return df[["review", "clean_review", "label", "category", "rating", "source"]]


def load_and_merge_datasets(args) -> pd.DataFrame:
    datasets = [
        prepare_dataframe(pd.read_csv(args.english_dataset), "english_reviews"),
        prepare_dataframe(pd.read_csv(args.tanglish_dataset), "tanglish_reviews"),
    ]

    if args.extra_dataset and Path(args.extra_dataset).exists():
        extra_df = prepare_dataframe(pd.read_csv(args.extra_dataset), "extra_tanglish_reviews")
        datasets.append(extra_df)

    df = pd.concat(datasets, ignore_index=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def build_common_preprocessor(max_text_features: int) -> ColumnTransformer:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=max_text_features,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=12000,
    )

    text_union = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        ("word_tfidf", word_tfidf),
                        ("char_tfidf", char_tfidf),
                    ]
                ),
            ),
            ("select", SafeSelectKBest(score_func=chi2, k=15000)),
        ]
    )

    numeric_columns = ["rating"]
    categorical_columns = ["category", "source"]

    return ColumnTransformer(
        transformers=[
            ("text", text_union, "clean_review"),
            ("stats", StructuredFeatureTransformer(), "review"),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_models(preprocessor: ColumnTransformer) -> dict:
    linear_feature_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scale", MaxAbsScaler()),
        ]
    )

    reduced_feature_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svd", TruncatedSVD(n_components=300, random_state=42)),
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("features", linear_feature_pipeline),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            steps=[
                ("features", linear_feature_pipeline),
                ("model", LinearSVC(class_weight="balanced", random_state=42)),
            ]
        ),
        "Multinomial Naive Bayes": Pipeline(
            steps=[
                ("features", linear_feature_pipeline),
                ("model", MultinomialNB(alpha=0.7)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("features", reduced_feature_pipeline),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    return models


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result = {
        "model": model_name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1_score": f1_score(y_test, test_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }
    return result, model


def try_save_visualizations(results_df: pd.DataFrame, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return False

    metric_columns = ["test_accuracy", "precision", "recall", "f1_score"]
    plot_df = results_df.set_index("model")[metric_columns]

    plt.figure(figsize=(10, 6))
    plot_df.plot(kind="bar")
    plt.title("Baseline Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300)
    plt.close()

    for _, row in results_df.iterrows():
        cm = np.array(row["confusion_matrix"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {row['model']}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        safe_name = row["model"].lower().replace(" ", "_")
        plt.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=300)
        plt.close()

    return True


def save_results(results_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "baseline_results.csv", index=False)

    confusion_rows = []
    for _, row in results_df.iterrows():
        cm = row["confusion_matrix"]
        confusion_rows.append(
            {
                "model": row["model"],
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1],
            }
        )
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)

    summary = {
        "best_model_by_f1": results_df.sort_values("f1_score", ascending=False).iloc[0]["model"],
        "best_model_by_accuracy": results_df.sort_values("test_accuracy", ascending=False).iloc[0]["model"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return try_save_visualizations(results_df, output_dir)


def evaluate_by_source(best_model, test_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    wrong_rows = []

    for source_name, source_df in test_df.groupby("source"):
        X_source = source_df[["review", "clean_review", "category", "rating", "source"]]
        y_true = source_df["label"]
        y_pred = best_model.predict(X_source)
        cm = confusion_matrix(y_true, y_pred).tolist()

        rows.append(
            {
                "source": source_name,
                "samples": int(len(source_df)),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1],
            }
        )

        source_errors = source_df.copy()
        source_errors["predicted_label"] = y_pred
        source_errors = source_errors[source_errors["label"] != source_errors["predicted_label"]].copy()
        source_errors["actual_label_name"] = source_errors["label"].map({0: "real", 1: "fake"})
        source_errors["predicted_label_name"] = source_errors["predicted_label"].map({0: "real", 1: "fake"})
        wrong_rows.append(
            source_errors[
                [
                    "source",
                    "review",
                    "clean_review",
                    "actual_label_name",
                    "predicted_label_name",
                ]
            ]
        )

        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        (output_dir / f"classification_report_{source_name}.txt").write_text(report, encoding="utf-8")

    source_results_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    source_results_df.to_csv(output_dir / "source_wise_results.csv", index=False)

    if wrong_rows:
        combined_errors = pd.concat(wrong_rows, ignore_index=True)
        combined_errors.to_csv(output_dir / "wrong_predictions_by_source.csv", index=False)

        tanglish_errors = combined_errors[combined_errors["source"] == "tanglish_reviews"].reset_index(drop=True)
        if not tanglish_errors.empty:
            tanglish_errors.to_csv(output_dir / "tanglish_wrong_predictions.csv", index=False)

    return source_results_df


def leakage_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    run_near_duplicate_check: bool = False,
    near_threshold: float = 0.92,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_clean = train_df[["source", "review", "clean_review", "label"]].copy().reset_index(drop=True)
    test_clean = test_df[["source", "review", "clean_review", "label"]].copy().reset_index(drop=True)

    exact_overlap = test_clean.merge(
        train_clean,
        on="clean_review",
        how="inner",
        suffixes=("_test", "_train"),
    )
    exact_overlap.to_csv(output_dir / "exact_train_test_overlap.csv", index=False)

    near_matches_df = pd.DataFrame()
    if run_near_duplicate_check:
        near_matches = []
        tanglish_test = test_clean[test_clean["source"] == "tanglish_reviews"].reset_index(drop=True)
        tanglish_train = train_clean[train_clean["source"] == "tanglish_reviews"].reset_index(drop=True)

        # Limit the expensive comparison to Tanglish reviews only.
        for _, test_row in tanglish_test.iterrows():
            test_text = test_row["clean_review"]
            if len(test_text) < 15:
                continue

            best_ratio = 0.0
            best_train_row = None

            for _, train_row in tanglish_train.iterrows():
                train_text = train_row["clean_review"]
                if abs(len(test_text) - len(train_text)) > 20:
                    continue
                if test_text[:12] != train_text[:12]:
                    continue

                ratio = SequenceMatcher(None, test_text, train_text).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_train_row = train_row

            if best_train_row is not None and best_ratio >= near_threshold:
                near_matches.append(
                    {
                        "source": "tanglish_reviews",
                        "similarity": round(best_ratio, 4),
                        "test_review": test_row["review"],
                        "test_clean_review": test_row["clean_review"],
                        "test_label": test_row["label"],
                        "train_review": best_train_row["review"],
                        "train_clean_review": best_train_row["clean_review"],
                        "train_label": best_train_row["label"],
                    }
                )

        near_matches_df = pd.DataFrame(near_matches)
    near_matches_df.to_csv(output_dir / "near_duplicate_train_test_overlap.csv", index=False)

    report = {
        "exact_overlap_count": int(len(exact_overlap)),
        "near_duplicate_overlap_count": int(len(near_matches_df)),
        "near_duplicate_threshold": near_threshold,
        "near_duplicate_check_ran": run_near_duplicate_check,
    }
    (output_dir / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def save_models(models: dict, trained_models: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except Exception as exc:
        print(f"joblib not available, skipping model persistence: {exc}")
        return

    for model_name in models:
        if model_name in trained_models:
            safe_name = model_name.lower().replace(" ", "_")
            joblib.dump(trained_models[model_name], output_dir / f"{safe_name}.joblib")


def run_baselines(df: pd.DataFrame, args):
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    X_train = train_df[["review", "clean_review", "category", "rating", "source"]]
    X_test = test_df[["review", "clean_review", "category", "rating", "source"]]
    y_train = train_df["label"]
    y_test = test_df["label"]

    preprocessor = build_common_preprocessor(args.max_text_features)
    models = build_models(preprocessor)

    results = []
    trained_models = {}

    for model_name, model in models.items():
        result, trained_model = evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        results.append(result)
        trained_models[model_name] = trained_model
        print(
            f"{model_name}: "
            f"accuracy={result['test_accuracy']:.4f}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1={result['f1_score']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False).reset_index(drop=True)
    plots_created = save_results(results_df, Path(args.output_dir))
    save_models(models, trained_models, Path(args.output_dir))
    best_model_name = results_df.iloc[0]["model"]
    source_results_df = evaluate_by_source(trained_models[best_model_name], test_df, Path(args.output_dir))

    leakage_report = leakage_check(
        train_df,
        test_df,
        Path(args.output_dir),
        run_near_duplicate_check=args.near_duplicate_check,
    )

    return results_df, source_results_df, leakage_report, plots_created, train_df, test_df


def run_transformer(train_df, test_df, args):
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        print(f"Transformer stage skipped because required packages are missing: {exc}")
        return None

    model_output_dir = Path(args.output_dir) / "muril_transformer"
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_model)

    train_dataset = Dataset.from_pandas(
        train_df[["review", "label"]].rename(columns={"review": "text"}), preserve_index=False
    )
    test_dataset = Dataset.from_pandas(
        test_df[["review", "label"]].rename(columns={"review": "text"}), preserve_index=False
    )

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=192)

    tokenized_train = train_dataset.map(tokenize_batch, batched=True)
    tokenized_test = test_dataset.map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.transformer_model, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.transformer_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    metrics_path = model_output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Transformer metrics saved to {metrics_path}")
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_merge_datasets(args)

    dataset_report = {
        "total_rows": int(df.shape[0]),
        "class_distribution": df["label"].value_counts().sort_index().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict(),
        "missing_rating": int(df["rating"].isna().sum()),
        "duplicate_free_rows": int(df.shape[0]),
    }
    Path(args.output_dir, "dataset_report.json").write_text(
        json.dumps(dataset_report, indent=2), encoding="utf-8"
    )
    print("Dataset report:")
    print(json.dumps(dataset_report, indent=2))

    results_df, source_results_df, leakage_report, plots_created, train_df, test_df = run_baselines(df, args)
    print("\nBaseline results:")
    print(results_df[["model", "test_accuracy", "precision", "recall", "f1_score"]].to_string(index=False))
    print("\nBest model source-wise results:")
    print(source_results_df.to_string(index=False))
    print("\nLeakage check:")
    print(json.dumps(leakage_report, indent=2))

    if plots_created:
        print("\nPlots saved in output directory.")
    else:
        print("\nMatplotlib or seaborn is not installed, so only CSV/JSON outputs were saved.")

    if args.run_transformer:
        run_transformer(train_df, test_df, args)


if __name__ == "__main__":
    main()

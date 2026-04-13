import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tanglish_fake_review_pipeline import load_and_merge_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer model for Tanglish fake review detection"
    )
    parser.add_argument(
        "--english-dataset",
        default=r"C:\Users\SRINITHI R\Downloads\ML\fake reviews dataset.csv",
    )
    parser.add_argument(
        "--tanglish-dataset",
        default=r"C:\Users\SRINITHI R\Downloads\ML\tanglish_2000_reviews.csv",
    )
    parser.add_argument("--extra-dataset", default="")
    parser.add_argument("--output-dir", default="dl_artifacts")
    parser.add_argument("--model-name", default="google/muril-base-cased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit(
            "Missing transformer dependencies. Install torch, transformers, and datasets first. "
            f"Original error: {exc}"
        )

    class DatasetArgs:
        pass

    dataset_args = DatasetArgs()
    dataset_args.english_dataset = args.english_dataset
    dataset_args.tanglish_dataset = args.tanglish_dataset
    dataset_args.extra_dataset = args.extra_dataset
    dataset_args.output_dir = str(output_dir)
    dataset_args.test_size = args.test_size
    dataset_args.random_state = args.random_state
    dataset_args.run_transformer = False
    dataset_args.transformer_model = args.model_name
    dataset_args.transformer_epochs = args.epochs
    dataset_args.max_text_features = 25000

    df = load_and_merge_datasets(dataset_args)

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = Dataset.from_pandas(
        train_df[["review", "label"]].rename(columns={"review": "text"}),
        preserve_index=False,
    )
    test_dataset = Dataset.from_pandas(
        test_df[["review", "label", "source"]].rename(columns={"review": "text"}),
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_train = train_dataset.map(tokenize_batch, batched=True)
    tokenized_test = test_dataset.map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "report_to": "none",
        "logging_steps": 25,
        "fp16": torch.cuda.is_available(),
    }

    try:
        training_args = TrainingArguments(
            eval_strategy="epoch",
            save_strategy="epoch",
            **training_kwargs,
        )
    except TypeError:
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            **training_kwargs,
        )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_test,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=1)],
    }

    try:
        trainer = Trainer(
            processing_class=tokenizer,
            **trainer_kwargs,
        )
    except TypeError:
        try:
            trainer = Trainer(
                tokenizer=tokenizer,
                **trainer_kwargs,
            )
        except TypeError:
            trainer = Trainer(**trainer_kwargs)

    trainer.train()
    eval_metrics = trainer.evaluate()

    predictions = trainer.predict(tokenized_test)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    source_df = test_df[["source"]].reset_index(drop=True).copy()
    source_df["true_label"] = test_df["label"].reset_index(drop=True)
    source_df["pred_label"] = pred_labels

    source_rows = []
    for source_name, group in source_df.groupby("source"):
        source_rows.append(
            {
                "source": source_name,
                "samples": int(len(group)),
                "accuracy": accuracy_score(group["true_label"], group["pred_label"]),
                "precision": precision_score(group["true_label"], group["pred_label"], zero_division=0),
                "recall": recall_score(group["true_label"], group["pred_label"], zero_division=0),
                "f1_score": f1_score(group["true_label"], group["pred_label"], zero_division=0),
            }
        )

    source_results_df = pd.DataFrame(source_rows)
    source_results_df.to_csv(output_dir / "transformer_source_wise_results.csv", index=False)
    pd.DataFrame(
        {
            "text": test_df["review"].reset_index(drop=True),
            "source": test_df["source"].reset_index(drop=True),
            "true_label": test_df["label"].reset_index(drop=True),
            "pred_label": pred_labels,
        }
    ).to_csv(output_dir / "transformer_predictions.csv", index=False)

    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    (output_dir / "transformer_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )

    print("Transformer evaluation metrics:")
    print(json.dumps(eval_metrics, indent=2))
    print("\nTransformer source-wise results:")
    print(source_results_df.to_string(index=False))


if __name__ == "__main__":
    main()

# Tanglish Fake Review Detection

This project trains baseline machine learning models and an optional transformer model for fake review detection on English and Tanglish code-mixed reviews.

## Files

- `tanglish_fake_review_pipeline.py`: end-to-end preprocessing, feature engineering, baseline training, evaluation, confusion matrices, and optional transformer fine-tuning
- `requirements.txt`: Python packages for the full pipeline

## Recommended approach

- Keep the baseline models for comparison
- Use both word and character TF-IDF instead of only word TF-IDF
- Add slang-aware normalization and handcrafted review features
- Fine-tune `google/muril-base-cased` for the final Tanglish-aware model

## Run baseline models

```powershell
python tanglish_fake_review_pipeline.py
```

## Run with an extra Tanglish dataset

```powershell
python tanglish_fake_review_pipeline.py --extra-dataset "C:\path\to\extra_tanglish_dataset.csv"
```

## Run the transformer model

```powershell
python tanglish_fake_review_pipeline.py --run-transformer
```

## Outputs

The script writes results into the `artifacts` folder:

- `dataset_report.json`
- `baseline_results.csv`
- `confusion_matrices.csv`
- `summary.json`
- saved model files
- optional plot images when `matplotlib` and `seaborn` are installed

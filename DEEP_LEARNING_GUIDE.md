# Deep Learning Step

This file explains the deep learning part of your rubric in beginner-friendly form.

## Why we need this

Your rubric asks for:

- implementation of a deep learning model
- comparison with baseline ML models
- justification for improved performance
- optimization techniques

Your project title also says **transformer-based**, so the deep learning model should be a transformer.

## Recommended model

Use **MuRIL**:

- model name: `google/muril-base-cased`
- reason: it works well for Indian language text and transliterated text such as Tanglish

## Files

- `tanglish_fake_review_pipeline.py`: baseline ML models
- `deep_learning_transformer.py`: transformer training script

## Easiest way to run

Because `torch` installation failed on your laptop, the easiest option is **Google Colab** for the deep learning stage.

## Colab steps

1. Open Google Colab.
2. Upload these two files:
   - `deep_learning_transformer.py`
   - `tanglish_fake_review_pipeline.py`
3. Upload your datasets:
   - `fake reviews dataset.csv`
   - `tanglish_2000_reviews.csv`
4. Run:

```python
!pip install pandas numpy scikit-learn datasets transformers torch
```

5. Run:

```python
!python deep_learning_transformer.py \
  --english-dataset "/content/fake reviews dataset.csv" \
  --tanglish-dataset "/content/tanglish_2000_reviews.csv"
```

## What the script gives you

Inside the `dl_artifacts` folder:

- `transformer_metrics.json`
- `transformer_source_wise_results.csv`
- `transformer_predictions.csv`
- saved transformer model in `best_model`

## How to compare with baseline

Compare these:

- baseline best model: Logistic Regression
- deep learning model: MuRIL transformer

Use:

- accuracy
- precision
- recall
- F1-score

## Optimization techniques you can mention

- pretrained transformer model
- early stopping
- tuned learning rate
- max sequence length setting
- batch-size tuning
- stratified train-test split

## Suggested report points

### Deep learning model implementation

We implemented a transformer-based deep learning model using MuRIL (`google/muril-base-cased`) for fake review detection in English and Tanglish code-mixed reviews.

### Why this model

MuRIL was selected because it is designed for Indian languages and transliterated text, making it suitable for Tanglish reviews where English and Tamil words are mixed.

### Comparison with baseline

The transformer model was compared against Logistic Regression, Linear SVM, Random Forest, and Multinomial Naive Bayes using accuracy, precision, recall, and F1-score.

### Justification for improved performance

Unlike TF-IDF based models, the transformer captures contextual meaning, slang usage, and spelling variations more effectively in code-mixed reviews.

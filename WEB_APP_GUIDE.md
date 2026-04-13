# Web App Guide

This guide explains what to do after your training is complete.

## What you already finished

- preprocessing
- feature engineering
- baseline model training
- Tanglish testing
- leakage check
- transformer training

## What this web app does

- takes typed review input
- takes voice review input from microphone
- converts voice to text in the browser
- sends the text to the Python backend
- predicts whether the review is real or fake

## Files

- `web_app.py`: Flask backend
- `templates/index.html`: frontend page with text box, mic button, and predict button

## Before running the web app

First make sure your baseline model exists:

- `artifacts/logistic_regression.joblib`

If it does not exist, run:

```powershell
python tanglish_fake_review_pipeline.py
```

## Install packages

```powershell
pip install flask joblib pandas
```

## Run the web app

```powershell
python web_app.py
```

## Open in browser

After running, open:

- `http://127.0.0.1:5000`

## How the microphone works

- the mic button is on the web page
- the browser asks for microphone permission
- speech is converted into text using the Web Speech API
- the text appears in the review box
- then you click predict

## Project flow

`Microphone -> Speech to Text -> Flask Backend -> Saved Model -> Prediction Result`

## Important note

For microphone support, use Chrome or Microsoft Edge.

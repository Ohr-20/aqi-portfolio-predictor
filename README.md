# US AQI Predictor - Vercel Deployment

This is the lightweight deployment package for the portfolio AQI project.

It uses:

- XGBoost regression for the numeric AQI prediction
- AQI category derived from the official AQI bands, ensuring it matches the predicted AQI
- A separate XGBoost classifier for the probability breakdown
- Manual location inputs: state, latitude, longitude, population, density, and reporting sites
- Pure-Python inference from exported model JSON files

The city-free model was validated on a chronological 200,000-row holdout. It achieved 79.39% category accuracy, 0.270 macro F1, and a 13.51 AQI mean absolute error.

## Deploy

Upload the contents of this folder to its own GitHub repository, then import that repository in Vercel.

Use:

```text
Framework Preset: Other
Install Command: leave empty
Build Command: leave empty
Output Directory: leave empty
```

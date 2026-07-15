# US AQI Predictor - Vercel Deployment

This is the lightweight deployment package for the portfolio AQI project.

It uses:

- XGBoost regression for the numeric AQI prediction
- AQI category derived from the official AQI bands, ensuring it matches the predicted AQI
- A separate XGBoost classifier for the probability breakdown
- City profiles that fill geographic and demographic inputs from the selected city
- Pure-Python inference from exported model JSON files

## Deploy

Upload the contents of this folder to its own GitHub repository, then import that repository in Vercel.

Use:

```text
Framework Preset: Other
Install Command: leave empty
Build Command: leave empty
Output Directory: leave empty
```

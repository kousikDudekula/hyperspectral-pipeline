# hyperspectral-pipeline

# Hyperspectral Image Processing Pipeline

## Overview
End-to-end pipeline processing NASA EMIT L1B 
hyperspectral satellite imagery with noise reduction,
PCA compression, and cloud integration.

## Pipeline Architecture
S3 Input (.nc file)
    → Load subset (200×200×150 bands)
    → Clean invalid values
    → Gaussian Denoising
    → SNR Calculation
    → PCA (150 bands → 3 components)
    → RGB + False Color Visualization
    → Upload results to S3
    → Flask API serves results

## Results
- Noise reduction across 150 spectral bands
- SNR improvement measured before vs after
- PCA compression: 150 bands → 3 components
- RGB and False Color images generated

## Tech Stack
Python | NumPy | SciPy | scikit-learn
NetCDF4 | Matplotlib | Flask | AWS S3 | boto3

## Setup
pip install -r requirements.txt

## Run Pipeline
python main.py

## API Endpoints
GET /health       → API status check
GET /process      → Trigger full pipeline
GET /snr-report   → Fetch SNR metrics

## Dataset
NASA EMIT L1B At-Sensor Calibrated Radiance
60m resolution | 285 spectral bands

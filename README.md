# Hyperspectral Image Processing Pipeline

An automated cloud-based pipeline for processing EMIT satellite hyperspectral data. Handles denoising, PCA-based dimensionality reduction, spectral visualization, and metric reporting — all triggered via a Flask REST API on AWS.

---

## Project Structure

```
├── main.py               # Pipeline orchestrator (download → process → upload)
├── app.py                # Flask REST API
├── src/
│   ├── load.py           # NetCDF4 data loading
│   ├── preprocess.py     # Data cleaning (invalid value removal)
│   ├── denoise.py        # Gaussian filtering + SNR calculation
│   ├── pca.py            # PCA dimensionality reduction
│   └── visualization.py  # RGB, false-color, and image saving
├── outputs/              # Generated images and metrics (local)
├── data/                 # Temporary storage for downloaded .nc files
└── requirements.txt
```

---

## How It Works

1. A `.nc` (NetCDF4) file is uploaded to S3 under the `input/` prefix
2. The `/process` endpoint is called with the filename
3. The pipeline downloads the file, processes it, and uploads results back to S3
4. Outputs are stored under a timestamped prefix: `output/<filename>_<timestamp>/`

### Processing Steps

```
Download from S3 → Load .nc → Clean (remove negatives) → Denoise (Gaussian)
→ Compute SNR & noise metrics → Apply PCA → Generate visualizations → Upload to S3
```

---

## API Endpoints

### `GET /health`
Check if the server is running.

**Response:**
```json
{"status": "running ✅"}
```

---

### `GET /process?file=<filename>`
Trigger the full processing pipeline for a given file.

**Query Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | string | `sample.nc` | Name of the `.nc` file in S3 `input/` |

**Response:**
```json
{
  "status": "completed",
  "result": {
    "snr": 26.1,
    "noise_before": 4.52,
    "noise_after": 0.43,
    "improvement": 90.5,
    "s3_output_path": "output/sample_20250101_120000"
  }
}
```

---

### `GET /snr-report`
Return the SNR metrics from the most recent pipeline run.

**Response:**
```json
{
  "status": "success",
  "report": {
    "SNR": "26.1",
    "Noise Before": "4.52",
    "Noise After": "0.43",
    "Improvement": "90.50%"
  }
}
```

---

## Outputs

| File | Description |
|------|-------------|
| `rgb.png` | True-color composite (Bands 28, 17, 7) |
| `false_color.png` | NIR false-color composite (Bands 100, 28, 17) |
| `pca.png` | First principal component (grayscale) |
| `denoise_comparison.png` | Side-by-side original vs. denoised (Band 50) |
| `snr.txt` | SNR, noise before/after, and improvement % |

---

## Tech Stack

- **Python**, **NumPy**, **scikit-learn**, **SciPy**, **Matplotlib**
- **Flask** — REST API
- **AWS S3** — input/output storage
- **AWS EC2** — pipeline execution
- **AWS Lambda** — event-driven trigger (S3 upload → pipeline)
- **NetCDF4** — satellite data format

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials
```bash
aws configure
```
Ensure the EC2 instance or environment has IAM permissions for `s3:GetObject` and `s3:PutObject` on the target bucket.

### 3. Run the Flask server
```bash
python app.py
```
Server starts on `http://0.0.0.0:5000`

### 4. Trigger a pipeline run
```bash
curl "http://localhost:5000/process?file=sample.nc"
```

---

## Metrics (Sample Run)

| Metric | Value |
|--------|-------|
| SNR | 26.1 |
| Noise Reduction | 90.5% |
| PCA Components | 3 |
| Dataset Size | Multi-GB NetCDF4 |
| Bands Processed | 150 |

---

## AWS Architecture

```
S3 (input/*.nc uploaded)
        ↓
    Lambda trigger
        ↓
  EC2 Flask API (/process)
        ↓
  Pipeline executes
        ↓
S3 (output/<name>_<timestamp>/)
```

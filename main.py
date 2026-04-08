from src.load import load_data
from src.preprocess import clean_data
from src.denoise import denoise, calculate_snr
from src.pca import apply_pca
from src.visualization import make_rgb, make_false_color, save_images

import matplotlib.pyplot as plt
import numpy as np
import os
import boto3
import datetime


BUCKET_NAME = "hyperspectral-data-koushik"


# =========================
# S3 DOWNLOAD
# =========================
def download_input(filename):
    s3 = boto3.client("s3")
    local_path = f"data/{filename}"

    try:
        print(f"[INFO] Downloading {filename} from S3...")
        s3.download_file(
            BUCKET_NAME,
            f"input/{filename}",
            local_path
        )
        return local_path

    except Exception as e:
        print("[ERROR] S3 download failed:", e)
        raise


# =========================
# S3 UPLOAD
# =========================
def upload_outputs(output_prefix):
    s3 = boto3.client("s3")

    files = [
        "rgb.png",
        "false_color.png",
        "pca.png",
        "denoise_comparison.png",
        "snr.txt"
    ]

    for file in files:
        try:
            s3.upload_file(
                f"outputs/{file}",
                BUCKET_NAME,
                f"{output_prefix}/{file}"
            )
            print(f"[INFO] Uploaded {file}")

        except Exception as e:
            print(f"[ERROR] Failed to upload {file}: {e}")


# =========================
# VISUALIZATION HELPERS
# =========================
def compare_denoising(original, denoised):
    band = min(50, original.shape[2] - 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original[:, :, band], cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(denoised[:, :, band], cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/denoise_comparison.png", dpi=150)
    plt.close()


def save_pca_image(pca_result):
    pca_img = pca_result[:, :, 0]

    p2 = np.nanpercentile(pca_img, 2)
    p98 = np.nanpercentile(pca_img, 98)

    pca_img = np.clip(
        (pca_img - p2) / (p98 - p2 + 1e-6),
        0,
        1
    )

    plt.imsave("outputs/pca.png", pca_img, cmap="gray")


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(filename="sample.nc"):

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    base_name = os.path.splitext(filename)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_prefix = f"output/{base_name}_{timestamp}"

    print("[INFO] Starting pipeline...")

    # -------------------------
    # Download Input
    # -------------------------
    local_file = download_input(filename)

    # -------------------------
    # Load & Clean
    # -------------------------
    print("[INFO] Loading data...")
    data = load_data(local_file)

    print("[INFO] Cleaning data...")
    data = clean_data(data)

    # -------------------------
    # Denoising
    # -------------------------
    print("[INFO] Denoising...")
    denoised = denoise(data)

    # -------------------------
    # Metrics
    # -------------------------
    print("[INFO] Calculating SNR...")
    snr = calculate_snr(data, denoised)

    noise_before = np.nanstd(data)
    noise_after = np.nanstd(data - denoised)

    improvement = (
        (noise_before - noise_after) / noise_before
    ) * 100

    print(f"[INFO] SNR: {snr}")
    print(f"[INFO] Noise Before: {noise_before:.2f}")
    print(f"[INFO] Noise After: {noise_after:.2f}")
    print(f"[INFO] Improvement: {improvement:.2f}%")

    # Save metrics
    with open("outputs/snr.txt", "w") as f:
        f.write(f"SNR: {snr}\n")
        f.write(f"Noise Before: {noise_before:.2f}\n")
        f.write(f"Noise After: {noise_after:.2f}\n")
        f.write(f"Improvement: {improvement:.2f}%\n")

    # -------------------------
    # PCA
    # -------------------------
    print("[INFO] Applying PCA...")
    pca_result = apply_pca(denoised)

    # -------------------------
    # Visualizations
    # -------------------------
    print("[INFO] Generating RGB & False Color...")
    rgb = make_rgb(denoised)
    fc = make_false_color(denoised)

    save_images(rgb, fc)

    print("[INFO] Saving PCA image...")
    save_pca_image(pca_result)

    print("[INFO] Saving denoise comparison...")
    compare_denoising(data, denoised)

    # -------------------------
    # Upload Results
    # -------------------------
    print("[INFO] Uploading results to S3...")
    upload_outputs(output_prefix)

    # -------------------------
    # Cleanup Local File
    # -------------------------
    if os.path.exists(local_file):
        os.remove(local_file)

    print("[INFO] Pipeline completed ✅")

    return {
        "snr": float(snr),
        "noise_before": float(noise_before),
        "noise_after": float(noise_after),
        "improvement": float(improvement),
        "s3_output_path": output_prefix
    }


if __name__ == "__main__":
    run_pipeline()
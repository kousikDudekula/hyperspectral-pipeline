from flask import Flask, jsonify, request
from main import run_pipeline
import traceback

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint to confirm the Flask server is running.

    Returns:
        JSON: {"status": "running ✅"}
    """
    return jsonify({"status": "running ✅"})


@app.route("/process", methods=["GET"])
def process():
    """
    Trigger the full hyperspectral processing pipeline for a given file.

    Query Params:
        file (str): Name of the .nc file in S3 'input/' prefix.
                    Defaults to 'sample.nc' if not provided.

    Returns:
        JSON: {
            "status": "completed",
            "result": {
                "snr": float,
                "noise_before": float,
                "noise_after": float,
                "improvement": float,
                "s3_output_path": str
            }
        }
        On failure:
        JSON: {"status": "error", "message": str}
    """
    # Read the target filename from query params, fall back to default
    filename = request.args.get("file", "sample.nc")

    try:
        result = run_pipeline(filename)

        # sklearn/numpy return numpy scalar types (e.g. np.float32) which
        # are not JSON-serializable by default — convert all values to
        # native Python types before passing to jsonify
        clean_result = {}
        if isinstance(result, dict):
            for k, v in result.items():
                try:
                    clean_result[k] = float(v)  # handles np.float32/64
                except (TypeError, ValueError):
                    clean_result[k] = v         # keep non-numeric values as-is (e.g. s3_output_path)
        else:
            clean_result = float(result)

        return jsonify({
            "status": "completed",
            "result": clean_result
        })

    except Exception as e:
        # Log full traceback server-side for debugging
        # Return only the exception message to the client
        print("[ERROR]", traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route("/snr-report", methods=["GET"])
def snr_report():
    """
    Read and return the most recently generated SNR metrics report.

    The report is read from 'outputs/snr.txt', which is written by
    run_pipeline() after each successful processing run. Each line
    is expected in 'Key: Value' format.

    Returns:
        JSON: {
            "status": "success",
            "report": {
                "SNR": str,
                "Noise Before": str,
                "Noise After": str,
                "Improvement": str
            }
        }
        If file not found:
        JSON: {"status": "error", "message": "No report found yet"}
    """
    try:
        with open("outputs/snr.txt", "r") as f:
            lines = f.readlines()

        # Parse each 'Key: Value' line into a dict
        # split(":", 1) ensures values containing ':' are not split
        report = {}
        for line in lines:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                report[key.strip()] = value.strip()

        return jsonify({
            "status": "success",
            "report": report
        })

    except Exception:
        # Most likely cause: pipeline hasn't run yet, so snr.txt doesn't exist
        return jsonify({
            "status": "error",
            "message": "No report found yet"
        })


if __name__ == "__main__":
    # Bind to all interfaces (0.0.0.0) so the server is reachable
    # from outside the container/EC2 instance, not just localhost
    app.run(host="0.0.0.0", port=5000)
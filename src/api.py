from flask import Flask, jsonify, request
from main import run_pipeline
import traceback

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running ✅"})


@app.route("/process", methods=["GET"])
def process():
    filename = request.args.get("file", "sample.nc")

    try:
        result = run_pipeline(filename)

        # ✅ Convert numpy types → Python types
        clean_result = {}
        if isinstance(result, dict):
            for k, v in result.items():
                try:
                    clean_result[k] = float(v)
                except:
                    clean_result[k] = v
        else:
            clean_result = float(result)

        return jsonify({
            "status": "completed",
            "result": clean_result
        })

    except Exception as e:
        print("[ERROR]", traceback.format_exc())

        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route("/snr-report", methods=["GET"])
def snr_report():
    try:
        with open("outputs/snr.txt", "r") as f:
            lines = f.readlines()

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
        return jsonify({
            "status": "error",
            "message": "No report found yet"
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
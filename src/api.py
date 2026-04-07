from flask import Flask, jsonify, request
from main import run_pipeline

app = Flask(__name__)

@app.route("/process", methods=["GET"])
def process():
    filename = request.args.get("file", "sample.nc")

    result = run_pipeline(filename)

    return jsonify({
        "status": "completed",
        "result": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
"""
Flask API server — bridges index.html frontend with the NLP matcher backend.

Run:
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from matcher import ResumeMatcher

app = Flask(__name__, static_folder=".")
CORS(app)

print("[INFO] Loading NLP model at startup...")
matcher = ResumeMatcher()
print("[INFO] Model ready. Server starting...\n")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        resume_file = request.files.get("resume")
        jd_file = request.files.get("jd")

        if not resume_file or not jd_file:
            return jsonify({"error": "Both 'resume' and 'jd' files are required."}), 400

        resume_text = resume_file.read().decode("utf-8", errors="ignore")
        jd_text = jd_file.read().decode("utf-8", errors="ignore")

        if not resume_text.strip() or not jd_text.strip():
            return jsonify({"error": "One or both files appear to be empty."}), 400

        result = matcher.score(resume_text, jd_text)

        # Ensure JSON-serializable (convert numpy floats)
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if hasattr(obj, 'item'):
                return obj.item()
            return obj

        return jsonify(clean(result))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  ResumeMatch API running at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)

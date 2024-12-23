import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace

# Disable oneDNN optimizations to avoid numerical differences or warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    try:
        # Check if 'frame' is in the request
        if 'frame' not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        # Read the frame from the request
        file = request.files['frame']
        file_bytes = file.read()

        # Decode the image
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Analyze the frame using DeepFace
        results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

        # Convert the results to Python-native types (floats, lists, etc.)
        results = convert_results(results)

        return jsonify(results), 200

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500

def convert_results(results):
    """Recursively converts numpy types to Python-native types."""
    if isinstance(results, list):
        return [convert_results(item) for item in results]
    elif isinstance(results, dict):
        return {key: convert_results(value) for key, value in results.items()}
    elif isinstance(results, np.generic):  # Handles numpy data types
        return results.item()  # Convert numpy types to native Python types (e.g., float32 -> float)
    else:
        return results

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


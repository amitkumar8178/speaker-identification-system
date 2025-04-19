from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
from predict import predict_speaker

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index5.html')


@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'audio' not in request.files:
        print("No audio in request")
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    temp_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
    audio_file.save(temp_path)
    print(f"Saved audio file to: {temp_path}")

    try:
        speaker = predict_speaker(temp_path)
        print(f"Prediction result: {speaker}")
        return jsonify({"speaker": speaker})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

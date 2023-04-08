import os
import json
import deepspeech
import numpy as np
import wave
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app, origins='http://localhost:3000')

@app.route('/api/speech-to-text', methods=['GET'])
@cross_origin()
def index():
    return jsonify('Welcome to DeepSpeechAPI!')

@app.route('/api/speech-to-text', methods=['POST'])
@cross_origin()
def speech_to_text():
    model_path = 'models/deepspeech-0.9.3-models.pbmm'
    scorer_path = 'models/deepspeech-0.9.3-models.scorer'
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'DeepSpeech model file not found.'}), 500
    
    if not os.path.exists(scorer_path):
        return jsonify({'error': 'DeepSpeech scorer file not found.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No audio file provided.'}), 400

    # Load the DeepSpeech model and scorer
    model = deepspeech.Model(model_path)
    model.enableExternalScorer(scorer_path)

    # Read the audio file and convert to 16-bit PCM format
    with wave.open(file, 'rb') as wav_file:
        rate = wav_file.getframerate()
        signal = wav_file.readframes(-1)
        pcm_data = np.frombuffer(signal, dtype=np.int16)
    
    # Transcribe the audio
    text = model.stt(pcm_data)

    # Get the confidence score
    # confidence = model.sttWithMetadata(pcm_data).confidences
    # Get confidence scores for each word

    # Transcribe audio with metadata
    metadata = model.sttWithMetadata(pcm_data)

    confidence = metadata.transcripts[0].confidence
    
    # # Calculate the average confidence score
    # avg_confidence = sum(confidences) / len(confidences)

    return jsonify({'text': text, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)

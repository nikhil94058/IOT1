from flask import Flask, request, jsonify
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the pre-trained model from the specified path
model_path = r'C:\Users\Hp\OneDrive\Desktop\sem4\coding\New folder\Nirbhaya\Nirbhaya\models\Emotion_Voice_Detection_Model.h5'  # Update with your model's path
model = load_model(model_path)

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess and predict audio input
def predict_audio_class(audio_data):
    # Load the audio file
    X, sr = librosa.load(audio_data, sr=16000)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
    
    # Reshape to match model input shape (1, 40, 1)
    mfccs = np.expand_dims(mfccs, axis=0)  # Shape (1, 40)
    mfccs = np.expand_dims(mfccs, axis=-1)  # Shape (1, 40, 1)

    # Predict class label
    predictions_probabilities = model.predict(mfccs)
    prediction = np.argmax(predictions_probabilities, axis=1)

    return prediction[0]

# Define API endpoint for audio predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an audio file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # Check if the file is an audio file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Process the audio file
    try:
        predicted_class = predict_audio_class(file)
        return jsonify({'predicted_class': int(predicted_class)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

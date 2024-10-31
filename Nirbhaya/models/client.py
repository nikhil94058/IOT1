import sounddevice as sd
import numpy as np
import wave
import requests
import time

# Mapping of predicted class labels to emotion descriptions
emotion_labels = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

def record_audio(duration=5, filename='temp_audio.wav'):
    print(f"Recording for {duration} seconds...")
    recorded_audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")
    
    # Save the recorded audio to a file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(16000)
        wf.writeframes((recorded_audio * 32767).astype(np.int16))  # Convert float to int16

    return filename

def send_audio_to_server(filename):
    url = 'http://localhost:5000/predict'  # Adjust to your server's URL
    with open(filename, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    return response.json()

if __name__ == '__main__':
    try:
        while True:
            audio_file = record_audio(duration=5)  # Record audio for 5 seconds
            prediction_result = send_audio_to_server(audio_file)
            predicted_class = prediction_result.get('predicted_class')

            # Map predicted class to emotion description
            emotion = emotion_labels.get(predicted_class, 'unknown')
            print(f"Predicted emotion for recorded audio: {emotion}")

            # Optional: Add a delay before the next recording
            time.sleep(1)  # Wait for 1 second before recording again
    except KeyboardInterrupt:
        print("Stopped recording.")

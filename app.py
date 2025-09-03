import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import librosa
import numpy as np

# --- Configuration (MUST MATCH train.py) ---
MODEL_PATH = "models/audio_classifier.pth"
SAMPLE_RATE = 8000
N_MFCC = 16
MAX_LEN_FRAMES = 256

# --- Re-define the Model Architecture (MUST be identical to training script) ---
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=10, max_len_frames=MAX_LEN_FRAMES):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        
        # The linear layer size must be updated for the new, deeper architecture
        self.fc = nn.Linear(128 * (N_MFCC // 8) * (max_len_frames // 8), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# --- Load the trained model AND normalization stats ---
device = torch.device("cpu")
model = AudioClassifier(num_classes=10).to(device)
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    mean = state['mean']
    std = state['std']
    model.eval()
    print("Model and normalization stats loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
    exit()
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("This usually means app.py's model architecture is out of sync with train.py.")
    exit()

# --- Flask App ---
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    tmp_path = ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        # 1. Load audio using librosa
        audio_data, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        
        # --- 2. ENHANCED FEATURE PIPELINE (Consistent with training) ---
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mfcc = librosa.feature.mfcc(S=mel_spectrogram_db, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        # 3. Pad or truncate (consistent with training)
        if mfcc.shape[1] < MAX_LEN_FRAMES:
            pad_width = MAX_LEN_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN_FRAMES]
        
        # 4. Apply the loaded normalization stats
        mfcc = (mfcc - mean) / std

        # 5. Convert to tensor, add channel and batch dimensions
        input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            prediction = predicted_idx.item()

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)


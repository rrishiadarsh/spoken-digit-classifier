import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio
import numpy as np
import os
import io
import librosa
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "audio_classifier.pth")
SAMPLE_RATE = 8000
N_MFCC = 16
BATCH_SIZE = 32
EPOCHS = 25 # Increased epochs for the more powerful model
LEARNING_RATE = 0.0005 
MAX_LEN_FRAMES = 256

# --- 1. Custom Dataset for Audio Processing (with Normalization) ---
class AudioDataset(Dataset):
    def __init__(self, hf_dataset, max_len, sample_rate, n_mfcc, mean=None, std=None):
        self.dataset = hf_dataset
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_bytes = item['audio']['bytes']
        
        audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True)
        
        # --- ENHANCED FEATURE PIPELINE ---
        # 1. Create a Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate, n_mels=128)
        # 2. Convert to Decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # 3. Calculate MFCCs from the dB-scaled spectrogram
        mfcc = librosa.feature.mfcc(S=mel_spectrogram_db, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        
        # Pad or truncate the TIME dimension
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]
        
        # Apply normalization if stats are provided
        if self.mean is not None and self.std is not None:
            mfcc = (mfcc - self.mean) / self.std

        # Add a channel dimension for the CNN and convert to tensor
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)
            
        return mfcc, label

# --- 2. A Deeper, More Powerful Model Architecture ---
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

# --- Main training and evaluation logic ---
def main():
    print("--- Loading Free Spoken Digit Dataset (FSDD) from mteb ---")
    fsdd_dataset = load_dataset("mteb/free-spoken-digit-dataset")
    fsdd_dataset = fsdd_dataset.cast_column("audio", Audio(decode=False))
    train_data = fsdd_dataset['train']
    test_data = fsdd_dataset['test']
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # --- NORMALIZE FEATURES ---
    print("\n--- Calculating normalization statistics from training data ---")
    pre_dataset = AudioDataset(train_data, MAX_LEN_FRAMES, SAMPLE_RATE, N_MFCC)
    pre_loader = DataLoader(pre_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    mean = 0.0
    std = 0.0
    num_samples = 0
    for inputs, _ in tqdm(pre_loader):
        batch_samples = inputs.size(0)
        inputs = inputs.view(batch_samples, inputs.size(2), -1)
        mean += inputs.mean(2).sum(0)
        std += inputs.std(2).sum(0)
        num_samples += batch_samples
    
    mean /= num_samples
    std /= num_samples
    mean = mean.unsqueeze(1).numpy()
    std = std.unsqueeze(1).numpy()
    print("Normalization stats calculated.")

    train_dataset = AudioDataset(train_data, MAX_LEN_FRAMES, SAMPLE_RATE, N_MFCC, mean=mean, std=std)
    test_dataset = AudioDataset(test_data, MAX_LEN_FRAMES, SAMPLE_RATE, N_MFCC, mean=mean, std=std)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AudioClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Model Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    print("--- Finished Training ---\n")

    print("--- Evaluating Model Performance ---")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    target_names = [str(i) for i in range(10)]
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    state = {
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std
    }
    torch.save(state, MODEL_PATH)
    print(f"Model and normalization stats saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()


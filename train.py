import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.model import WakeWordCNN
from utils.audio_processing import wav_to_mel, fix_spectrogram

def train_wake_word_model(data_dir, epochs=20, batch_size=32, patience=5):
    """Train the wake word detector model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and process data
    print("Loading audio files...")
    marvinler = []
    basqalari = []
    
    # Load marvin samples
    marvin_dir = os.path.join(data_dir, "marvin")
    if os.path.exists(marvin_dir):
        for file in os.listdir(marvin_dir):
            if file.endswith(".wav"):
                mel = wav_to_mel(os.path.join(marvin_dir, file))
                marvinler.append(mel)
    
    # Load other word samples (limit to 2100)
    for folder in os.listdir(data_dir):
        if folder != "marvin" and os.path.isdir(os.path.join(data_dir, folder)):
            folder_path = os.path.join(data_dir, folder)
            count = 0
            for file in os.listdir(folder_path):
                if file.endswith(".wav") and count < 2100 // (len(os.listdir(data_dir)) - 1):
                    mel = wav_to_mel(os.path.join(folder_path, file))
                    basqalari.append(mel)
                    count += 1
    
    print(f"Marvin samples: {len(marvinler)}")
    print(f"Other samples: {len(basqalari)}")
    
    # Fix spectrogram sizes
    for i in range(len(marvinler)):
        marvinler[i] = fix_spectrogram(marvinler[i])
    for i in range(len(basqalari)):
        basqalari[i] = fix_spectrogram(basqalari[i])
    
    # Prepare dataset
    marvinler = np.array(marvinler)
    basqalari = np.array(basqalari)
    
    X = np.concatenate([marvinler, basqalari], axis=0)
    y = np.concatenate([np.ones(len(marvinler)), np.zeros(len(basqalari))], axis=0)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"Total samples: {len(X)}")
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train[:, None, :, :], dtype=torch.float32)
    X_val = torch.tensor(X_val[:, None, :, :], dtype=torch.float32)
    X_test = torch.tensor(X_test[:, None, :, :], dtype=torch.float32)
    
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Model, loss, optimizer
    model = WakeWordCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
        
        avg_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)
        
        val_acc = correct_val / total_val
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            correct_test += (preds == y_batch).sum().item()
            total_test += y_batch.size(0)
    
    test_acc = correct_test / total_test
    print(f"\nTest Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    data_dir = "."  # Directory containing wake word folders
    train_wake_word_model(data_dir)

# Wake Word Detector

A deep learning-based wake word detection system that identifies specific audio keywords in real-time using Mel-spectrogram analysis and PyTorch neural networks.

## 📋 Overview

This project implements an intelligent audio classification system that listens for specific wake words in continuous audio streams. It uses librosa for audio processing and PyTorch for neural network inference, converting raw audio into Mel-spectrograms for classification.

## ✨ Features

- **Real-time Detection**: Process audio streams with sliding window approach
- **Mel-Spectrogram Analysis**: Converts raw audio into frequency-based features optimized for speech
- **Flexible Configuration**: Customizable audio parameters (sample rate, mel bands, time steps)
- **Robust Inference**: Adjustable confidence thresholds and hit counting for reliable detection
- **GPU Support**: Leverages CUDA when available for faster processing
- **Normalized Audio Processing**: Automatic audio standardization and padding

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShahinGanbar/Wake-Word-Detector-local.git
   cd Wake_Word_detector
   ```

2. Install dependencies:
   ```bash
   pip install librosa numpy torch
   ```

3. Or use requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## 📚 Project Structure

```
Wake_Word_detector/
├── assets/
├── app.py                 # Streamlit web interface
├── train.py              # Model training script
├── config.py             # Configuration constants
├── best_model.pth        # Trained neural network weights
├── requirements.txt      # Python dependencies
├── utils/
│   ├── audio_processing.py    # Audio to mel-spectrogram conversion
│   └── model.py              # Neural network architecture
└── README.md
```

## 🚀 Quick Start

### Basic Usage

```python
import numpy as np
from utils.audio_processing import detect_wake_word
import torch

# Load your trained model
model = torch.load('models/your_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load or create audio (16kHz mono)
audio = np.random.randn(16000).astype(np.float32)  # 1 second

# Detect wake word
detected = detect_wake_word(
    audio=audio,
    model=model,
    device=device,
    threshold=0.85,
    min_hits=2
)

print(f"Wake word detected: {detected}")
```

## 📖 API Reference

### `wav_to_mel(path, sr=16000, n_mels=64)`

Converts a WAV file to a standardized Mel-spectrogram for training.

**Parameters:**
- `path` (str): Path to WAV file
- `sr` (int): Sample rate in Hz (default: 16000)
- `n_mels` (int): Number of mel frequency bands (default: 64)

**Returns:**
- `np.ndarray`: Normalized Mel-spectrogram (n_mels × time_steps)

**Example:**
```python
from utils.audio_processing import wav_to_mel

mel_spec = wav_to_mel('audio/wake_word.wav', sr=16000, n_mels=64)
print(mel_spec.shape)  # (64, 31)
```

---

### `fix_spectrogram(mel, n_mels=64, time_steps=32)`

Normalizes spectrogram dimensions by padding or cropping to fixed time steps.

**Parameters:**
- `mel` (np.ndarray): Mel-spectrogram array
- `n_mels` (int): Number of mel bands (default: 64)
- `time_steps` (int): Target time dimension (default: 32)

**Returns:**
- `np.ndarray`: Fixed-size Mel-spectrogram (n_mels × time_steps)

**Example:**
```python
from utils.audio_processing import fix_spectrogram

mel_spec = fix_spectrogram(mel, n_mels=64, time_steps=32)
print(mel_spec.shape)  # (64, 32)
```

---

### `detect_wake_word(audio, model, device, sr=16000, n_mels=64, time_steps=32, hop_frames=4, threshold=0.85, min_hits=2)`

Detects wake word in audio using a trained model with sliding window inference.

**Parameters:**
- `audio` (np.ndarray): Audio samples as numpy array (mono, float32)
- `model` (torch.nn.Module): Trained PyTorch model
- `device` (torch.device): CPU or GPU device
- `sr` (int): Sample rate in Hz (default: 16000)
- `n_mels` (int): Number of mel bands (default: 64)
- `time_steps` (int): Spectrogram time dimension (default: 32)
- `hop_frames` (int): Frame stride for sliding window (default: 4)
- `threshold` (float): Confidence threshold 0-1 (default: 0.85)
- `min_hits` (int): Consecutive detections required (default: 2)

**Returns:**
- `bool`: `True` if wake word detected, `False` otherwise

**Example:**
```python
from utils.audio_processing import detect_wake_word
import torch

model = torch.load('models/wake_word.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

audio = np.random.randn(16000).astype(np.float32)
detected = detect_wake_word(audio, model, device, threshold=0.85, min_hits=2)
```

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sr` | 16000 | Sample rate in Hz |
| `n_mels` | 64 | Number of mel frequency bands |
| `n_fft` | 1024 | FFT window size |
| `hop_length` | 512 | Samples between frames |
| `time_steps` | 32 | Spectrogram time dimension |
| `hop_frames` | 4 | Frame stride for sliding window |
| `threshold` | 0.85 | Confidence threshold (0-1) |
| `min_hits` | 2 | Minimum consecutive detections |

## 🔧 Tuning Guide

### For Better Accuracy
- **Increase `min_hits`**: Requires more consecutive detections (e.g., 3-5)
- **Increase `threshold`**: Higher confidence requirement (e.g., 0.90-0.95)
- **Reduce `hop_frames`**: More windows to evaluate (slower but more thorough)

### For Better Speed
- **Decrease `min_hits`**: Fewer required detections (e.g., 1)
- **Decrease `threshold`**: Lower confidence requirement (faster but more false positives)
- **Increase `hop_frames`**: Larger frame stride (e.g., 8-16)

### Audio Quality Tips
- Use **16kHz mono** audio for best results
- Ensure audio is normalized (peak amplitude ≈ 1.0)
- Test with varying noise levels and distances

## 📋 Audio Processing Pipeline

1. **Loading**: Audio loaded at specified sample rate
2. **Mel-Spectrogram**: Raw audio converted to frequency domain
3. **Power-to-dB**: Spectrogram converted to decibel scale for better feature representation
4. **Normalization**: Features standardized by mean and std deviation
5. **Sliding Window**: Multiple time windows extracted for detection
6. **Inference**: Each window classified by neural network
7. **Hit Counting**: Consecutive high-confidence predictions aggregated

## ⚠️ Requirements

- **Minimum audio length**: 0.25 seconds (4000 samples at 16kHz)
- **Audio format**: Mono, 16-bit PCM recommended
- **Sample rate**: 16kHz (configurable, but training data should match)
- **GPU**: Optional but recommended for real-time processing

## 📦 Dependencies

```
librosa>=0.9.0
numpy>=1.21.0
torch>=1.9.0
```

See `requirements.txt` for complete dependency list.

## 🎯 Model Requirements

Your PyTorch model should:
- Accept input shape: `(batch_size, 1, n_mels, time_steps)` e.g., `(1, 1, 64, 32)`
- Output logits for 2 classes: `[negative_class, wake_word_class]`
- Support `.eval()` mode for inference

## 🐛 Troubleshooting

**Issue**: High false positive rate
- **Solution**: Increase `threshold` to 0.90+, increase `min_hits` to 3+

**Issue**: Missing detections
- **Solution**: Decrease `threshold` to 0.75, decrease `min_hits` to 1

**Issue**: Slow detection
- **Solution**: Increase `hop_frames` to 8-16, use GPU with CUDA

**Issue**: Audio length errors
- **Solution**: Ensure audio is at least 0.25 seconds long

## 📝 Notes

- Audio is padded or cropped to 1-second chunks in `wav_to_mel()`
- Mel-spectrograms use power-to-dB conversion with reference amplitude
- Sliding window approach allows detection in continuous audio streams
- Model runs in evaluation mode (`model.eval()`) to disable dropout/batch norm

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT


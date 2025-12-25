# utils/audio_processing.py
import librosa
import numpy as np
import torch

def wav_to_mel(path, sr=16000, n_mels=64):
    """Convert WAV file to Mel-spectrogram."""
    audio, sr = librosa.load(path, sr=sr)
    # Ensure all audio is 1 second for training consistency
    if len(audio) > sr:
        audio = audio[:sr]
    else:
        audio = np.pad(audio, (0, max(0, sr - len(audio))))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=1.0)

    # Standardize
    mel_db = (mel_db - (-40.0)) / 20.0
    return mel_db

def fix_spectrogram(mel, n_mels=64, time_steps=32):
    """Pad or crop Mel-spectrogram to fixed time steps."""
    current_steps = mel.shape[1]

    if current_steps < time_steps:
        mel = np.pad(mel, ((0, 0), (0, time_steps - current_steps)), mode='constant')
    else:
        mel = mel[:, :time_steps]

    return mel

def detect_wake_word(
    audio,
    model,
    device,
    sr=16000,
    n_mels=64,
    time_steps=32,
    hop_frames=4,
    threshold=0.85,
    min_hits=2
):
    """
    Detect wake word from audio array (numpy).
    audio: numpy array of audio samples
    Returns: bool indicating if wake word was detected
    """
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Return False if audio is too short
    if len(audio) < sr // 4:  # Less than 0.25 seconds
        return False
    
    # Convert to mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    model.eval()
    hits = 0

    # Sliding window detection
    if mel.shape[1] < time_steps:
        # Single window if mel is shorter than time_steps
        windows_to_process = [mel]
    else:
        windows_to_process = []
        for start in range(0, mel.shape[1] - time_steps + 1, hop_frames):
            windows_to_process.append(mel[:, start:start + time_steps])
    
    for window in windows_to_process:
        # Pad if necessary
        if window.shape[1] < time_steps:
            window = np.pad(window, ((0, 0), (0, time_steps - window.shape[1])), mode='constant')
        
        # Normalize
        window_std = window.std() + 1e-6
        window = (window - window.mean()) / window_std

        window_tensor = torch.tensor(
            window[None, None, :, :],
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(window_tensor), dim=1)
            wake_prob = probs[0, 1].item()

        if wake_prob >= threshold:
            hits += 1
            if hits >= min_hits:
                return True
        else:
            hits = 0

    return False

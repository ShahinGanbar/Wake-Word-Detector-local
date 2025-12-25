# app.py
import streamlit as st
import sounddevice as sd
import numpy as np
from utils.model import WakeWordCNN
from utils.audio_processing import detect_wake_word
import torch
import os
import threading
import queue

# Page config
st.set_page_config(page_title="Wake Word Detector", layout="centered")

# Device and model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WakeWordCNN().to(device)
    
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    else:
        st.warning("Model file 'best_model.pth' not found. Please train the model first.")
    
    model.eval()
    return model, device

model, device = load_model()

# Streamlit UI
st.title("🎤 Wake Word Detector - 'marvin'")
st.write("Microphone listens continuously for the wake word 'marvin'")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.5, 1.0, 0.85, step=0.05)
chunk_duration = st.sidebar.slider("Audio Chunk Duration (ms)", 500, 2000, 1000, step=100)
sensitivity = st.sidebar.slider("Min Consecutive Hits", 1, 5, 2)

# Create containers for status
status_container = st.empty()
result_container = st.empty()
stats_container = st.container()

# Audio processing parameters
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * chunk_duration / 1000)
BUFFER_SIZE = CHUNK_SIZE * 4  # Keep last 4 chunks in memory

def audio_callback(indata, frames, time_info, status_info):
    """Callback for audio stream."""
    if status_info:
        pass  # Handle status info if needed
    audio_queue.put(indata.copy().flatten())

# Start continuous listening
if st.button("🎙️ Start Listening", use_container_width=True, key="start_btn"):
    status_container.info("🔴 Microphone is listening...")
    
    audio_queue = queue.Queue()
    audio_buffer = np.array([], dtype=np.float32)
    detection_count = 0
    chunk_count = 0
    
    try:
        with st.spinner("🔴 Listening for wake word 'marvin'..."):
            # Start audio stream
            with sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            ):
                # Placeholder for real-time updates
                placeholder = st.empty()
                stats_placeholder = st.empty()
                
                while True:
                    try:
                        # Get audio chunk
                        chunk = audio_queue.get(timeout=2)
                        audio_buffer = np.concatenate([audio_buffer, chunk])
                        
                        # Keep buffer size manageable
                        if len(audio_buffer) > BUFFER_SIZE:
                            audio_buffer = audio_buffer[-BUFFER_SIZE:]
                        
                        chunk_count += 1
                        
                        # Process every 2 chunks
                        if chunk_count % 2 == 0 and len(audio_buffer) >= CHUNK_SIZE * 2:
                            # Check if audio has content
                            if np.max(np.abs(audio_buffer)) > 0.01:
                                # Normalize
                                audio_norm = audio_buffer / (np.max(np.abs(audio_buffer)) + 1e-8)
                                
                                # Detect wake word
                                detected = detect_wake_word(
                                    audio_norm,
                                    model,
                                    device,
                                    threshold=threshold,
                                    min_hits=sensitivity
                                )
                                
                                if detected:
                                    detection_count += 1
                                    placeholder.success(f"✨ **WAKE WORD DETECTED: 'marvin'** ✨ (Hit #{detection_count})")
                                else:
                                    detection_count = 0
                                    placeholder.info(f"🔴 Listening... (Chunk #{chunk_count})")
                            else:
                                placeholder.warning("🔊 No audio detected - speak louder")
                            
                            # Display statistics
                            with stats_placeholder.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Audio Chunks", chunk_count)
                                with col2:
                                    st.metric("Detections", detection_count)
                                with col3:
                                    st.metric("Max Amplitude", f"{np.max(np.abs(audio_buffer)):.3f}")
                    
                    except queue.Empty:
                        placeholder.warning("⚠️ No audio input detected")
                        break
                    except KeyboardInterrupt:
                        break
    
    except Exception as e:
        status_container.error(f"❌ Error: {str(e)}")
        st.info("Make sure your microphone is connected and you granted permissions.")

else:
    status_container.info("Click 'Start Listening' to begin monitoring for 'marvin'")

st.divider()
st.subheader("💡 How to use:")
st.write("""
1. Click **Start Listening** to activate the microphone
2. Speak the word **'marvin'** clearly
3. Watch the status updates in real-time
4. Adjust the threshold if detections are too sensitive/insensitive
5. The microphone will continuously listen until you stop the script
""")

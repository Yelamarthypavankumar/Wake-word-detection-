import streamlit as st
import openwakeword
import numpy as np
import sounddevice as sd
import time
import os
from collections import deque
import pygame

# ---------------- CONFIG ----------------
WAKEWORD_FOLDER = "wake_words"
INFERENCE_FRAMEWORK = "onnx"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
CONFIDENCE_THRESHOLD = 0.5
SILENCE_FRAMES = 8
DISPLAY_DURATION = 2  # seconds to show detection message

# Initialize pygame mixer once
pygame.mixer.init()

def play_detection_sound():
    try:
        pygame.mixer.music.load("ding.wav")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Failed to play sound: {e}")

st.title("🎤 Wake Word Detection")

# Dynamically list all ONNX models in wake_words folder
if not os.path.isdir(WAKEWORD_FOLDER):
    st.error(f"Wake word folder '{WAKEWORD_FOLDER}' not found.")
    st.stop()

model_files = [f for f in os.listdir(WAKEWORD_FOLDER) if f.endswith(".onnx")]

if not model_files:
    st.error(f"No ONNX models found in '{WAKEWORD_FOLDER}' folder.")
    st.stop()

selected_model_file = st.selectbox("Choose a wake word model:", model_files)
start_button = st.button("▶️ Start Listening")
status_box = st.empty()

if start_button:
    model_path = os.path.join(WAKEWORD_FOLDER, selected_model_file)

    # Load the model once
    try:
        owwModel = openwakeword.Model(
            wakeword_models=[model_path],
            inference_framework=INFERENCE_FRAMEWORK
        )
        st.success(f"✅ Loaded model: {selected_model_file}")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    wakeword_buffer = deque(maxlen=SILENCE_FRAMES)
    last_detection_time = 0

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype='float32'
        ) as stream:
            while True:
                audio_block, _ = stream.read(CHUNK_SIZE)
                audio = audio_block[:, 0]
                audio_int16 = (audio * 32767).astype(np.int16)

                prediction = owwModel.predict(audio_int16)
                confidence = list(prediction.values())[0]
                print(f"Wake word confidence: {confidence:.2f}")

                wakeword_buffer.append(confidence >= CONFIDENCE_THRESHOLD)
                now = time.time()

                # Play sound and update last detection time on new detection
                if any(wakeword_buffer):
                    if now - last_detection_time > DISPLAY_DURATION:
                        play_detection_sound()
                    last_detection_time = now

                # Show detection message for DISPLAY_DURATION seconds after last detection
                if now - last_detection_time <= DISPLAY_DURATION:
                    status_box.success(f"🔊 Wake word detected! ({selected_model_file})")
                else:
                    status_box.info(f"🎤 Listening for: {selected_model_file}...")

                time.sleep(0.1)

    except Exception as e:
        st.error(f"❌ Audio stream error: {e}")


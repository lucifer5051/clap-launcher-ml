import os
import sys
import time
import collections
import pickle
import numpy as np
import pyaudio
import librosa
import subprocess
import webbrowser

# Load the trained 95%+ precision machine learning model
try:
    with open("clap_model.pkl", "rb") as f:
        clf = pickle.load(f)
except FileNotFoundError:
    print("Error: clap_model.pkl not found! Please run train_model.py first.")
    sys.exit(1)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WINDOW_CHUNKS = int(RATE * 0.6 / CHUNK)  # ~0.6 seconds of rolling audio buffer
CONFIDENCE_THRESHOLD = 0.70

MIN_CLAP_GAP = 0.01
MAX_CLAP_GAP = 0.3
COOLDOWN_AFTER_FIRE = 2.0

def launch_systems():
    print("[JARVIS] Double clap confirmed! (>70% confidence). Launching systems in Chrome...")
    
    try:
        # Force Google Chrome specifically via Windows Shell to direct video URL to autoplay
        if os.system('start chrome "https://www.youtube.com/watch?v=pAgnJDJN4VA&list=RDpAgnJDJN4VA&start_radio=1"') != 0:
            raise Exception
    except Exception:
        try:
            webbrowser.open("https://www.youtube.com/watch?v=pAgnJDJN4VA&list=RDpAgnJDJN4VA&start_radio=1")
        except Exception:
            pass
        
    time.sleep(2.0)
    
    try:
        subprocess.Popen(["antigravity"])
    except Exception:
        try:
            if os.system("start antigravity") != 0:
                raise Exception
        except Exception:
            try:
                webbrowser.open("https://antigravity.app")
            except Exception:
                pass
                
    print("[JARVIS] Done.")

def extract_features_from_buffer(audio_buffer):
    # Flatten the list of numpy chunks into a single 1D float32 array
    # We must divide by 32768.0 to get range [-1.0, 1.0] to exactly mirror librosa.load() behavior
    y = np.concatenate(audio_buffer).astype(np.float32) / 32768.0 
    
    # Needs to match exactly the features in train_model.py
    mfccs = librosa.feature.mfcc(y=y, sr=RATE, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    rms = np.mean(librosa.feature.rms(y=y)[0])
    
    features = np.hstack([mfccs_mean, mfccs_std, zcr, rms])
    return features.reshape(1, -1)

def main():
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Audio Initialization Error: {e}")
        sys.exit(1)

    print("[JARVIS] ML Clap detector online. Listening...")

    # Set up our rolling audio buffer so the ML model can see backward in time
    buffer = collections.deque(maxlen=WINDOW_CHUNKS)
    for _ in range(WINDOW_CHUNKS):
        buffer.append(np.zeros(CHUNK, dtype=np.int16))
        
    last_clap_time = 0
    cooldown_until = 0

    try:
        while True:
            # Shift buffer
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            buffer.append(audio_data)
            
            # Basic volume tripwire -> bypasses CPU heavy ML inference if room is silent
            rms_volume = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
            current_time = time.time()
            
            if rms_volume > 250 and current_time > cooldown_until: 
                # Spike detected, run the audio through the ML matrix
                X = extract_features_from_buffer(buffer)
                
                probs = clf.predict_proba(X)[0]
                clap_prob = probs[1] # Probability of class 1 (Clap)
                
                if clap_prob >= CONFIDENCE_THRESHOLD:
                    print(f"-> Single Clap Detected | Confidence: {clap_prob*100:.1f}%")
                    
                    time_since_last = current_time - last_clap_time
                    
                    if MIN_CLAP_GAP <= time_since_last <= MAX_CLAP_GAP:
                        # Success Window!
                        launch_systems()
                        
                        last_clap_time = 0
                        cooldown_until = time.time() + COOLDOWN_AFTER_FIRE
                        
                        # Clear old audio that built up while sleeping
                        try:
                            stream.read(stream.get_read_available(), exception_on_overflow=False)
                        except:
                            pass
                        
                        # Zero out sliding memory buffer to prevent hallucination double triggers
                        for _ in range(WINDOW_CHUNKS):
                            buffer.append(np.zeros(CHUNK, dtype=np.int16))
                        continue
                    else:
                        last_clap_time = current_time

                    # Force a micro cooldown to prevent classifying the trailing tail-end of the same clap
                    cooldown_until = max(cooldown_until, current_time + 0.01)
                    
            # Reset timeline if you clap once and walk away
            if last_clap_time > 0 and (current_time - last_clap_time) > 1.5:
                last_clap_time = 0

    except KeyboardInterrupt:
        print("\n[JARVIS] Shutting down...")
    finally:
        try:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        except:
            pass

if __name__ == "__main__":
    main()

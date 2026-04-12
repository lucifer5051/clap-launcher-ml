import os
import sys
import time
import subprocess
import webbrowser
import collections

try:
    import pyaudio
    import numpy as np
except ImportError:
    print("ERROR: Missing required libraries.")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━
# TUNABLE SETTINGS
# ━━━━━━━━━━━━━━━━━━
CLAP_SENSITIVITY     = 3.0    # higher = less sensitive
MIN_CLAP_GAP         = 0.3    # seconds minimum between two claps
MAX_CLAP_GAP         = 0.8    # seconds maximum between two claps
COOLDOWN_AFTER_FIRE  = 2.0    # seconds before it listens again
RESET_AFTER          = 1.5    # seconds to reset if no second clap
MOVING_AVG_WINDOW    = 30     # chunks used for dynamic threshold

# ━━━━━━━━━━━━━━━━━━
# AUDIO SETTINGS
# ━━━━━━━━━━━━━━━━━━
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def launch_systems():
    print("[JARVIS] Double clap detected! Launching systems...")
    
    # Step 1: Open YouTube
    try:
        webbrowser.open("https://www.youtube.com/results?search_query=back+in+black+acdc")
    except Exception:
        pass
        
    # Step 2: Wait 0.5s
    time.sleep(0.5)
    
    # Step 3: Launch Antigravity
    try:
        # Try a)
        subprocess.Popen(["antigravity"])
    except Exception:
        try:
            # Try b)
            if os.system("start antigravity") != 0:
                raise Exception("os.system failed")
        except Exception:
            try:
                # Try c)
                webbrowser.open("https://antigravity.app")
            except Exception:
                pass
                
    print("[JARVIS] Done.")

def main():
    try:
        audio = pyaudio.PyAudio()
    except Exception as e:
        print(f"Error initializing PyAudio: {e}")
        sys.exit(1)

    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Error opening microphone stream: {e}")
        sys.exit(1)

    print("[JARVIS] Clap detector online. Listening...")

    # Initialize circular buffer for moving average
    energy_buffer = collections.deque(maxlen=MOVING_AVG_WINDOW)
    # Pre-fill buffer with nominal noise floor
    for _ in range(MOVING_AVG_WINDOW):
        energy_buffer.append(100.0)
    
    last_clap_time = 0
    
    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
            except Exception as e:
                time.sleep(0.01)
                continue
                
            # Calculate RMS energy of each chunk using numpy
            rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
            
            # Dynamic threshold = mean(buffer) * CLAP_SENSITIVITY
            moving_average = np.mean(energy_buffer)
            threshold = moving_average * CLAP_SENSITIVITY
            
            if threshold < 100: 
                threshold = 100
                
            current_time = time.time()
                
            if rms > threshold and rms > 300: 
                # Potential clap detected
                if current_time - last_clap_time > 0.1: # Debounce extremely fast peaks
                    time_since_last = current_time - last_clap_time
                    
                    if MIN_CLAP_GAP <= time_since_last <= MAX_CLAP_GAP:
                        # DOUBLE CLAP CONFIRMED -> trigger action
                        launch_systems()
                        
                        last_clap_time = 0
                        time.sleep(COOLDOWN_AFTER_FIRE)
                        
                        # Reset buffer after fire
                        for _ in range(MOVING_AVG_WINDOW):
                            energy_buffer.append(100.0)
                        
                        # Clear old audio left in buffer during wait
                        try:
                            stream.read(stream.get_read_available(), exception_on_overflow=False)
                        except:
                            pass
                            
                        continue # Start fresh listening cycle
                    else:
                        # Record timestamp as first clap_event
                        last_clap_time = current_time
            
            # Reset if no second clap in time
            if last_clap_time > 0 and (current_time - last_clap_time) > RESET_AFTER:
                last_clap_time = 0
                
            # Update circular buffer of last N RMS values
            energy_buffer.append(float(rms))
            
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

import os
import sys
import wave
import glob
import pyaudio
import numpy as np
import collections
import sounddevice as sd

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
ENERGY_THRESHOLD = 200
RECORD_SECONDS = 1.0

def play_audio(frames):
    # Flatten frames and play natively using sounddevice
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    sd.play(audio_data, RATE)
    sd.wait()

def save_wav(filename, frames):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def get_highest_index(directory, prefix):
    files = glob.glob(os.path.join(directory, f"{prefix}_*.wav"))
    if not files:
        return 0
    nums = []
    for f in files:
        try:
            num = int(os.path.basename(f).split('_')[1].split('.')[0])
            nums.append(num)
        except:
            pass
    return max(nums) if nums else 0

def draw_progress(claps, noises, target=60):
    c_perc = min(int((claps / target) * 20), 20)
    n_perc = min(int((noises / target) * 20), 20)
    
    c_bar = ("=" * (c_perc - 1) + ">").ljust(20) if c_perc > 0 else " " * 20
    if claps >= target: c_bar = "=" * 20
        
    n_bar = ("=" * (n_perc - 1) + ">").ljust(20) if n_perc > 0 else " " * 20
    if noises >= target: n_bar = "=" * 20
        
    print(f"  Claps  [{c_bar}] {claps}/{target}")
    print(f"  Noise  [{n_bar}] {noises}/{target}")

def main():
    base_dir = "C:\\Users\\digvi\\.gemini\\antigravity\\scratch\\clap_launcher\\dataset"
    claps_dir = os.path.join(base_dir, "claps")
    noise_dir = os.path.join(base_dir, "noise")
    
    os.makedirs(claps_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    
    claps_count = get_highest_index(claps_dir, "clap")
    noise_count = get_highest_index(noise_dir, "noise")
    skipped_count = 0
    history = []  # Stack of (type, filepath)

    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"[JARVIS] Audio Initialization Error: {e}")
        sys.exit(1)

    os.system('cls' if os.name == 'nt' else 'clear')
    print("=========================================")
    print("      JARVIS MANUAL LABELING MODULE      ")
    print("=========================================")
    print(f"Threshold: {ENERGY_THRESHOLD} RMS | Duration: {RECORD_SECONDS}s")
    print("Commands: y (clap) | n (noise) | skip | undo | done")
    print("=========================================")

    try:
        while True:
            if claps_count >= 60 and noise_count >= 60:
                print("\n")
                draw_progress(claps_count, noise_count)
                print("\n[JARVIS] Dataset complete, Sir. Run train_model.py next.")
                break
                
            print("\n[LISTENING] Ready for next sound...")
            
            # Pre-roll buffer captures fraction of a second BEFORE threshold hits
            # This ensures the sharp peak (attack) of a clap isn't cut off
            rolling_buffer = collections.deque(maxlen=3) # ~0.06s pre-buffer
            
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                rolling_buffer.append(data)
                
                audio_data = np.frombuffer(data, dtype=np.int16)
                # Quick volume calculation
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                
                if rms > ENERGY_THRESHOLD:
                    break
                    
            print("[JARVIS] Sound anomaly detected. Recording 1-sec clip...")
            frames = list(rolling_buffer)
            
            chunks_to_read = int(RATE / CHUNK * RECORD_SECONDS) - len(frames)
            for _ in range(chunks_to_read):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
            play_audio(frames)
            
            while True:
                choice = input(">> Was this a CLAP? (y/n/skip): ").strip().lower()
                
                if choice == 'done':
                    print("\n[JARVIS] Shutting down manual labeling offline.")
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return
                    
                if choice == 'undo':
                    if not history:
                        print("[JARVIS] Memory clear. No previous action to undo.")
                        continue
                        
                    last_type, last_path = history.pop()
                    if os.path.exists(last_path):
                        os.remove(last_path)
                        
                    if last_type == 'clap': claps_count -= 1
                    elif last_type == 'noise': noise_count -= 1
                    
                    print(f"[JARVIS] Understood. Erased previous '{last_type}'.")
                    break
                    
                if choice in ['y', 'yes']:
                    claps_count += 1
                    filename = os.path.join(claps_dir, f"clap_{claps_count:03d}.wav")
                    save_wav(filename, frames)
                    history.append(('clap', filename))
                    break
                    
                elif choice in ['n', 'no']:
                    noise_count += 1
                    filename = os.path.join(noise_dir, f"noise_{noise_count:03d}.wav")
                    save_wav(filename, frames)
                    history.append(('noise', filename))
                    break
                    
                elif choice in ['s', 'skip']:
                    skipped_count += 1
                    break
                    
                else:
                    print("[JARVIS] Invalid directive. Valid: y, n, skip, undo, done")
                    
            # Clear microphone buffer to prevent instantly cross-triggering 
            # off background noise that occurred during typing.
            try:
                stream.read(stream.get_read_available(), exception_on_overflow=False)
            except:
                pass
                
            if choice not in ['undo', 'done']:
                print(f"[SAVED] Claps: {claps_count} | Noise: {noise_count} | Skipped: {skipped_count}")
                draw_progress(claps_count, noise_count)
                
    except KeyboardInterrupt:
        print("\n[JARVIS] Keyboard interrupt offline.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except:
            pass

if __name__ == "__main__":
    main()

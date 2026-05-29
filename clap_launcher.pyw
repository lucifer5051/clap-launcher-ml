"""
clap_launcher.pyw
=================
Silent background script: detects double and triple clap sounds and opens specific websites in Google Chrome.
Run with pythonw.exe so no console window appears.

Dependencies: pyaudio, numpy
Install: pip install pyaudio numpy
"""

# ============================================================
#  USER-CONFIGURABLE VARIABLES  (edit these freely)
# ============================================================

# The exact path to the Google Chrome executable
CHROME_PATH = "C:/Program Files/Google/Chrome/Application/chrome.exe"

# URLs for DOUBLE CLAP - each will open in its own tab
DOUBLE_CLAP_URLS = [
    "https://claude.ai/new",
    "https://chatgpt.com/",
    "https://notebooklm.google.com/",
    "https://gemini.google.com/",
]

# URL for TRIPLE CLAP
TRIPLE_CLAP_URLS = [
    "https://youtu.be/IyR25B-IGyg?si=y4DaxZF9PEvXBGev",
]

# How loud a single "chunk" must be to even be considered a clap candidate.
# Range: 0.0 – 1.0  (higher = less sensitive, lower = more sensitive)
CLAP_THRESHOLD = 0.25

# How fast the volume must RISE (attack) to count as a clap.
# This is the minimum ratio of current RMS to the previous chunk's RMS.
# Range: 1.5 – 10.0  (higher = stricter, fewer false positives)
ATTACK_RATIO = 4.0

# How quickly the volume must FALL after the attack (decay ratio).
# The RMS of the chunk AFTER the peak must be below this fraction of the peak.
# Range: 0.1 – 0.8  (lower = stricter)
DECAY_RATIO = 0.6

# gap timing requirements
DOUBLE_CLAP_GAP = 0.8  # Max time between 2 claps for a Double Clap
TRIPLE_CLAP_GAP = 1.2  # Max time between 3 claps for a Triple Clap

# Cooldown gap before the script can trigger AGAIN (prevents accidental spamming)
COOLDOWN = 15.0

# Audio capture settings
SAMPLE_RATE = 44100
CHUNK_SIZE  = 1024        # samples per audio frame (~23 ms)
CHANNELS    = 1


# ============================================================
#  CORE LOGIC — modify only if you know what you're doing
# ============================================================

import time
import subprocess
import threading
import os
import sys
import ctypes

import numpy as np
import pyaudio

# ── Suppress all stdout/stderr output (runs silently as .pyw) ──────────────
try:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
except Exception:
    pass

# ── Prevent Windows from going to sleep ──────────────────────────────────────
try:
    ES_CONTINUOUS      = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
except Exception:
    pass


def rms(data: np.ndarray) -> float:
    """Root-mean-square of a float32 audio chunk."""
    return float(np.sqrt(np.mean(data ** 2)))


def open_chrome_urls(urls):
    """Open specified URLs in Google Chrome."""
    args = [CHROME_PATH] + urls
    try:
        subprocess.Popen(args)
    except Exception:
        pass


def execute_double_clap():
    threading.Thread(target=open_chrome_urls, args=(DOUBLE_CLAP_URLS,), daemon=True).start()


def execute_triple_clap():
    threading.Thread(target=open_chrome_urls, args=(TRIPLE_CLAP_URLS,), daemon=True).start()


def listen():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    prev_rms      = 0.0       
    prev_prev_rms = 0.0       

    clap_times    = []        
    last_trigger  = 0.0       

    try:
        while True:
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.float32)
            current_rms = rms(samples)

            now = time.monotonic()

            # ── Clap conditions ────────────────────────────────────────────
            is_loud_enough  = current_rms > CLAP_THRESHOLD
            has_fast_attack = (prev_rms > 0) and (current_rms / prev_rms >= ATTACK_RATIO)
            has_fast_decay  = (prev_prev_rms < current_rms * DECAY_RATIO)

            if is_loud_enough and has_fast_attack and has_fast_decay:
                clap_times.append(now)

                # Exactly 3 claps
                if len(clap_times) == 3:
                    if (clap_times[2] - clap_times[0]) <= TRIPLE_CLAP_GAP:
                        if (now - last_trigger) >= COOLDOWN:
                            execute_triple_clap()
                            last_trigger = now
                        clap_times = []  # reset
                    else:
                        # Oldest clap is too old, remove it and keep the newest two 
                        clap_times.pop(0)

            # Evaluate claps if window is expiring without a 3rd clap
            if clap_times:
                time_since_first = now - clap_times[0]
                
                # If TRIPLE_CLAP_GAP time has elapsed since the first clap, 
                # we know a valid triple clap cannot be formed anymore.
                if time_since_first > TRIPLE_CLAP_GAP:
                    if len(clap_times) == 2:
                        # Validate double clap constraint
                        if (clap_times[1] - clap_times[0]) <= DOUBLE_CLAP_GAP:
                            if (now - last_trigger) >= COOLDOWN:
                                execute_double_clap()
                                last_trigger = now
                    
                    # Reset the window
                    clap_times = []

            prev_prev_rms = prev_rms
            prev_rms      = current_rms

    except Exception:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    listen()

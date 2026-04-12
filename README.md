# 👏 JARVIS ML Clap Launcher

An ultra-responsive, Machine Learning powered background daemon that detects natural double-claps to trigger desktop events—completely immune to background noise and keyboard clatter!

By default, the script will silently listen in the background and instantly open the AC/DC "Back in Black" music video when you clap twice, followed by launching the Antigravity desktop application. 

## 🚀 Features
* **AI Powered**: Uses `scikit-learn` and `librosa` MFCC audio extraction to understand the mathematical shape of a true clap, easily ignoring desk bumps, coughs, and heavy mechanical keyboard typing.
* **Insanely Fast**: Calibrated to recognize rapid double-claps with less than 0.01 seconds required between claps.
* **Pre-Trained Brain**: Comes pre-loaded with a highly robust `clap_model.pkl` model honed against heavy ambient noise, meaning it works brilliantly out of the box with zero training required.
* **Low CPU Overhead**: Uses a mathematical basic volume tripwire (`rms > 250`) to sleep the system until a loud noise actually occurs, ensuring your CPU isn't wasting cycles running ML inference on silence.

---

## 🛠️ Installation & Setup (Any Windows PC)

1. **Clone the Repo**
   ```bash
   git clone https://github.com/lucifer5051/JARVIS-Clap-Launcher.git
   cd JARVIS-Clap-Launcher
   ```

2. **Install Dependencies**
   Ensure you have Python 3.12+ installed, then run:
   ```bash
   pip install pyaudio numpy scikit-learn librosa sounddevice
   ```

3. **Run the Daemon!**
   Simply run the main launcher:
   ```bash
   python ml_clap_launcher.py
   ```
   *(It will load the pre-trained `clap_model.pkl` brain immediately and seamlessly start listening in the background!)*

---

# 👏 JARVIS ML Clap Launcher

An ultra-responsive, Machine Learning powered background daemon that detects natural double-claps to trigger desktop events—completely immune to background noise and keyboard clatter!

By default, the script will silently listen in the background and instantly open the AC/DC "Back in Black" music video when you clap twice, followed by launching the Antigravity desktop application. 

## 🚀 Features
* **AI Powered**: Uses `scikit-learn` and `librosa` MFCC audio extraction to understand the mathematical shape of a true clap, easily ignoring desk bumps, coughs, and heavy mechanical keyboard typing.
* **Insanely Fast**: Calibrated to recognize rapid double-claps with less than 0.01 seconds required between claps.
* **Built in Trainer UI**: Comes with a fully standalone offline HTML5 Web Audio interface (`trainer_ui.html`) allowing anyone to easily record and label their own audio dataset to train a custom room-specific model!
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
   *(It will load the pre-trained `clap_model.pkl` brain immediately and start listening!)*

---

## 🎧 Want to train your own custom model?
If the pre-trained model gives you false positives for your specific room, you can rebuild the AI entirely from scratch in 5 minutes!

1. Open `trainer_ui.html` in Google Chrome or Microsoft Edge.
2. The browser will ask for microphone permission. Click **Allow**.
3. Record ~100 **Claps**, and ~100 ambient **Noise** clips (type on your keyboard, bump your desk, talk, etc).
4. Click **Download Dataset ZIP** on the UI.
5. Extract the downloaded `/dataset/` folder into the exact same folder as the scripts.
6. Run the training pipeline:
   ```bash
   python train_model.py
   ```
   This will completely overwrite `clap_model.pkl` with your own highly specific acoustics! Run `ml_clap_launcher.py` and it will use your new brain.

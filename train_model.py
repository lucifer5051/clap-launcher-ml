import os
import glob
import pickle
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Constants for spike isolation
WINDOW_SEC = 0.15 # 150ms window around the spike
SR = 44100

def get_feature_vector(y, sr):
    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Extract Zero Crossing Rate for hit detection
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    
    # Extract RMS Energy
    rms = np.mean(librosa.feature.rms(y=y)[0])
    
    # Extract Spectral characteristics (specifically hyper-targets Human Voice formants)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(centroid)
    cent_std = np.std(centroid)
    
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    band_mean = np.mean(bandwidth)
    band_std = np.std(bandwidth)

    # Combine spatial/frequency characteristics into a single flat vector
    features = np.hstack([mfccs_mean, mfccs_std, zcr, rms, cent_mean, cent_std, band_mean, band_std])
    return features

def isolate_spike(y, sr):
    """Finds the absolute peak and returns a WINDOW_SEC clip around it."""
    peak_idx = np.argmax(np.abs(y))
    half_win = int((WINDOW_SEC / 2) * sr)
    
    start = max(0, peak_idx - half_win)
    end = min(len(y), peak_idx + half_win)
    
    # Pad if necessary to keep consistent feature vector size
    spike = y[start:end]
    if len(spike) < half_win * 2:
        spike = np.pad(spike, (0, (half_win * 2) - len(spike)), mode='constant')
        
    return spike

def extract_background(y, sr):
    """Takes segments from the beginning and end of the file, assuming they are noise."""
    win_len = int(WINDOW_SEC * sr)
    
    # Get first segment and last segment
    lead_in = y[:win_len]
    if len(lead_in) < win_len:
        lead_in = np.pad(lead_in, (0, win_len - len(lead_in)), mode='constant')
        
    lead_out = y[-win_len:]
    if len(lead_out) < win_len:
        lead_out = np.pad(lead_out, (0, win_len - len(lead_out)), mode='constant')
        
    return lead_in, lead_out

def main():
    print("Initiating Impulse-Focused Spike Training...")
    
    X = []
    y_labels = []
    
    claps = glob.glob("dataset/claps/*.wav")
    
    if len(claps) == 0:
        print("Error: Could not find any clap files in dataset/claps/.")
        return
        
    print(f"Found {len(claps)} clap recordings. Isolating spikes and background air...")
        
    for f in claps:
        y, sr = librosa.load(f, sr=SR)
        
        # 1. Extract the Spike (Class 1)
        spike = isolate_spike(y, sr)
        X.append(get_feature_vector(spike, sr))
        y_labels.append(1)
        
        # 2. Extract Background Noise (Class 0) from the SAME file
        bg1, bg2 = extract_background(y, sr)
        X.append(get_feature_vector(bg1, sr))
        y_labels.append(0)
        X.append(get_feature_vector(bg2, sr))
        y_labels.append(0)
        
        # Augment the spike slightly (5x) to make it bulletproof
        for _ in range(4):
            noise_amp = np.random.uniform(0.001, 0.01)
            spike_aug = spike + noise_amp * np.random.randn(len(spike))
            X.append(get_feature_vector(spike_aug, sr))
            y_labels.append(1)

    X = np.array(X)
    y_labels = np.array(y_labels)
    
    print(f"Dataset generated: {len(X)} samples total.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.15, random_state=42, stratify=y_labels)
    
    print("\nTraining Impulse-Classifier (GradientBoosting)...")
    clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.08, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy on Test Split: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Background", "Clap Spike"]))
    
    # Save the model
    model_path = "clap_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"\n[SUCCESS] Spike-Focused model saved to {model_path}.")

if __name__ == "__main__":
    main()

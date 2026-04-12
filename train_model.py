import os
import glob
import pickle
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def extract_features(file_path):
    # Load audio. sr=44100 ensures we don't downsample. 
    # librosa automatically scales to float32 between -1.0 and 1.0!
    y, sr = librosa.load(file_path, sr=44100)
    
    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Extract Zero Crossing Rate for hit detection
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    
    # Extract RMS Energy
    rms = np.mean(librosa.feature.rms(y=y)[0])
    
    # Combine spatial/frequency characteristics into a single flat vector
    features = np.hstack([mfccs_mean, mfccs_std, zcr, rms])
    return features

def main():
    print("Loading audio files and extracting features... This may take a minute.")
    
    X = []
    y_labels = []
    
    claps = glob.glob("dataset/claps/*.wav")
    noises = glob.glob("dataset/noise/*.wav")
    
    if len(claps) == 0 or len(noises) == 0:
        print("Error: Could not find dataset files. Did you run collect_data.py?")
        return
        
    print(f"Found {len(claps)} claps and {len(noises)} noises.")
        
    # Process Claps
    for f in claps:
        X.append(extract_features(f))
        y_labels.append(1) # 1 = Clap
        
    # Process Noises
    for f in noises:
        X.append(extract_features(f))
        y_labels.append(0) # 0 = Noise
        
    X = np.array(X)
    y_labels = np.array(y_labels)
    
    # Stratified Split guarantees both test and train sets get balance
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    
    print("\nTraining RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluation target: >95%
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy on Test Split: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Noise (0)", "Clap (1)"]))
    
    if acc < 0.95:
        print("\n[WARNING] Accuracy is below the 95% target. If it gives false flags later, record more diverse noise!")
    else:
        print("\n[SUCCESS] Target >95% accuracy reached.")
        
    # Save the model artifact
    model_path = "clap_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"\nModel saved to {model_path}. Proceed to running ml_clap_launcher.py!")

if __name__ == "__main__":
    main()

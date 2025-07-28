import os
import librosa # For audio processing
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import argparse

# GPU Configuration
def setup_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} device(s) available")
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration failed: {e}")
            return False
    else:
        print("⚠️ No GPU detected, using CPU")
        return False

# Setup GPU on import
setup_gpu()

# --- Configuration ---
# Path to the audio datasets
AUDIO_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'audio')
REAL_AUDIO_PATH = os.path.join(AUDIO_DATASET_PATH, 'real')
FAKE_AUDIO_PATH = os.path.join(AUDIO_DATASET_PATH, 'fake')

# Path to load/save the trained audio model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'deepfake_audio_detector_model.h5')

# Model parameters
SAMPLE_RATE = 16000 # Standard sample rate for speech
DURATION = 5 # Process 5 seconds of audio
N_MFCC = 20 # Number of MFCC features to extract
EPOCHS = 20
BATCH_SIZE = 32

# --- Training Functions ---

def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast')
        
        # Pad or truncate the audio to the desired duration
        if len(audio) < DURATION * SAMPLE_RATE:
            audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)), 'constant')
        else:
            audio = audio[:DURATION * SAMPLE_RATE]

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        # We need to scale the features for the model
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_audio_dataset():
    """Loads audio files, extracts features, and assigns labels."""
    print("Loading audio dataset and extracting features...")
    features = []
    labels = []

    # Process REAL audio
    for filename in os.listdir(REAL_AUDIO_PATH):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            path = os.path.join(REAL_AUDIO_PATH, filename)
            mfccs = extract_features(path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(0) # 0 for Real

    # Process FAKE audio
    for filename in os.listdir(FAKE_AUDIO_PATH):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            path = os.path.join(FAKE_AUDIO_PATH, filename)
            mfccs = extract_features(path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(1) # 1 for Fake

    if not features:
        print("❌ Error: No audio files found. Please populate the 'datasets/audio' folder.")
        return None, None

    features = np.array(features)
    labels = np.array(labels)
    print(f"✅ Dataset loaded: {len(features)} audio files processed.")
    return features, labels

def build_audio_model(input_shape):
    """Builds a simple 1D CNN for audio classification."""
    print("Building the 1D CNN audio model...")
    model = Sequential([
        # We need to reshape the input for the 1D CNN
        tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print("✅ Audio model built and compiled.")
    return model

# --- Prediction Function ---

def predict_single_audio(file_path, model):
    """Predicts if a single audio file is real or fake."""
    print(f"\n--- Predicting audio: {os.path.basename(file_path)} ---")
    
    features = extract_features(file_path)
    if features is None:
        return
        
    features = np.expand_dims(features, axis=0) # Add batch dimension

    prediction = model.predict(features)[0][0]
    
    confidence = prediction * 100
    if prediction > 0.5:
        print(f"Result: ❌ DEEPFAKE AUDIO DETECTED")
        print(f"Confidence: {confidence:.2f}%")
    else:
        confidence = (1 - prediction) * 100
        print(f"Result: ✅ AUTHENTIC AUDIO (REAL)")
        print(f"Confidence: {confidence:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deepfake Audio Detector: Train or Predict.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'])
    parser.add_argument('--audio', type=str, help="Path to the audio file for prediction.")
    
    args = parser.parse_args()

    if args.mode == 'predict':
        if not args.audio:
            print("❌ Error: For 'predict' mode, you must provide an audio path using --audio.")
        elif not os.path.exists(MODEL_PATH):
            print(f"❌ Error: Trained model not found at {MODEL_PATH}. Run in 'train' mode first.")
        else:
            print(f"Loading trained audio model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            predict_single_audio(args.audio, model)

    elif args.mode == 'train':
        features, labels = load_audio_dataset()
        if features is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            model = build_audio_model(X_train.shape[1:])

            print("\n--- Starting Audio Model Training ---")
            model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      batch_size=BATCH_SIZE, epochs=EPOCHS)
            print("--- Audio Model Training Finished ---")

            print(f"\nSaving trained audio model to: {MODEL_PATH}")
            model.save(MODEL_PATH)
            print("✅ Audio model saved successfully!")

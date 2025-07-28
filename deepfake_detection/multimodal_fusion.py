#!/usr/bin/env python3
"""
Multimodal Fusion for Deepfake Detection
Combines image and audio detection results for improved accuracy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
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

# Configuration
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'faces')
AUDIO_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'audio')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
IMG_SIZE = 299
SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 20

class MultimodalFusionDetector:
    """Multimodal fusion detector combining image and audio analysis"""
    
    def __init__(self):
        """Initialize the multimodal fusion detector"""
        self.image_model = None
        self.audio_model = None
        self.fusion_model = None
        self.feature_extractor = None
        
        # Load models
        self._load_models()
        
        # Fusion weights (can be tuned)
        self.image_weight = 0.6
        self.audio_weight = 0.4
        
    def _load_models(self):
        """Load pre-trained image and audio models"""
        print("📥 Loading models...")
        
        # Load image detection model
        image_model_path = os.path.join(MODEL_PATH, 'deepfake_detector_model.h5')
        if os.path.exists(image_model_path):
            self.image_model = load_model(image_model_path)
            print("✅ Image model loaded")
        else:
            print("❌ Image model not found")
            
        # Load audio detection model
        audio_model_path = os.path.join(MODEL_PATH, 'deepfake_audio_detector_model.h5')
        if os.path.exists(audio_model_path):
            self.audio_model = load_model(audio_model_path)
            print("✅ Audio model loaded")
        else:
            print("❌ Audio model not found")
            
        # Load fusion model if exists
        fusion_model_path = os.path.join(MODEL_PATH, 'multimodal_fusion_model.pkl')
        if os.path.exists(fusion_model_path):
            with open(fusion_model_path, 'rb') as f:
                self.fusion_model = pickle.load(f)
            print("✅ Fusion model loaded")
        else:
            print("ℹ️ No fusion model found, using weighted average")
    
    def extract_image_features(self, image_path):
        """Extract features from image using pre-trained model"""
        if self.image_model is None:
            return None
            
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image / 255.0  # Normalize
            
            # Get features from intermediate layer
            feature_model = tf.keras.Model(
                inputs=self.image_model.input,
                outputs=self.image_model.layers[-2].output
            )
            
            features = feature_model.predict(np.expand_dims(image, axis=0), verbose=0)
            return features.flatten()
            
        except Exception as e:
            print(f"❌ Error extracting image features: {e}")
            return None
    
    def extract_audio_features(self, audio_path):
        """Extract MFCC features from audio"""
        if self.audio_model is None:
            return None
            
        try:
            # Load audio and extract MFCC
            audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast')
            
            # Pad or truncate
            if len(audio) < DURATION * SAMPLE_RATE:
                audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)), 'constant')
            else:
                audio = audio[:DURATION * SAMPLE_RATE]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            
            return mfccs_scaled
            
        except Exception as e:
            print(f"❌ Error extracting audio features: {e}")
            return None
    
    def predict_image(self, image_path):
        """Predict deepfake probability for image"""
        if self.image_model is None:
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image / 255.0
            
            prediction = self.image_model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting image: {e}")
            return None
    
    def predict_audio(self, audio_path):
        """Predict deepfake probability for audio"""
        if self.audio_model is None:
            return None
            
        try:
            features = self.extract_audio_features(audio_path)
            if features is None:
                return None
                
            features = np.expand_dims(features, axis=0)
            prediction = self.audio_model.predict(features, verbose=0)[0][0]
            return prediction
            
        except Exception as e:
            print(f"❌ Error predicting audio: {e}")
            return None
    
    def fuse_predictions(self, image_pred, audio_pred):
        """Fuse image and audio predictions"""
        if image_pred is None and audio_pred is None:
            return None, "No predictions available"
            
        if image_pred is None:
            # Only audio available
            confidence = audio_pred * 100 if audio_pred > 0.5 else (1 - audio_pred) * 100
            label = "DEEPFAKE" if audio_pred > 0.5 else "AUTHENTIC"
            return audio_pred, f"{label} (Audio only) | Confidence: {confidence:.2f}%"
            
        if audio_pred is None:
            # Only image available
            confidence = image_pred * 100 if image_pred > 0.5 else (1 - image_pred) * 100
            label = "DEEPFAKE" if image_pred > 0.5 else "AUTHENTIC"
            return image_pred, f"{label} (Image only) | Confidence: {confidence:.2f}%"
        
        # Both modalities available - use fusion
        if self.fusion_model is not None:
            # Use trained fusion model
            combined_features = np.concatenate([image_pred, audio_pred])
            fused_pred = self.fusion_model.predict_proba([combined_features])[0][1]
        else:
            # Use weighted average
            fused_pred = (self.image_weight * image_pred + self.audio_weight * audio_pred)
        
        confidence = fused_pred * 100 if fused_pred > 0.5 else (1 - fused_pred) * 100
        label = "DEEPFAKE" if fused_pred > 0.5 else "AUTHENTIC"
        
        return fused_pred, f"{label} (Multimodal) | Confidence: {confidence:.2f}%"
    
    def detect_deepfake(self, image_path=None, audio_path=None):
        """Main detection function combining image and audio analysis"""
        print("\n🔍 Multimodal Deepfake Detection")
        print("=" * 50)
        
        image_pred = None
        audio_pred = None
        
        # Process image if provided
        if image_path and os.path.exists(image_path):
            print(f"📸 Analyzing image: {os.path.basename(image_path)}")
            image_pred = self.predict_image(image_path)
            if image_pred is not None:
                img_conf = image_pred * 100 if image_pred > 0.5 else (1 - image_pred) * 100
                img_label = "DEEPFAKE" if image_pred > 0.5 else "AUTHENTIC"
                print(f"   Image Result: {img_label} | Confidence: {img_conf:.2f}%")
        
        # Process audio if provided
        if audio_path and os.path.exists(audio_path):
            print(f"🎵 Analyzing audio: {os.path.basename(audio_path)}")
            audio_pred = self.predict_audio(audio_path)
            if audio_pred is not None:
                aud_conf = audio_pred * 100 if audio_pred > 0.5 else (1 - audio_pred) * 100
                aud_label = "DEEPFAKE" if audio_pred > 0.5 else "AUTHENTIC"
                print(f"   Audio Result: {aud_label} | Confidence: {aud_conf:.2f}%")
        
        # Fuse predictions
        fused_pred, result = self.fuse_predictions(image_pred, audio_pred)
        
        print(f"\n🎯 Final Result: {result}")
        
        return fused_pred, result
    
    def train_fusion_model(self, image_data_dir, audio_data_dir):
        """Train the fusion model on combined image and audio data"""
        print("🚀 Training Multimodal Fusion Model")
        print("=" * 50)
        
        # This would require paired image-audio data
        # For now, we'll use the weighted average approach
        print("ℹ️ Using weighted average fusion (image: 60%, audio: 40%)")
        print("💡 To train a custom fusion model, provide paired image-audio datasets")

def main():
    """Main function for multimodal detection"""
    parser = argparse.ArgumentParser(description="Multimodal Deepfake Detection")
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--train', action='store_true', help='Train fusion model')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MultimodalFusionDetector()
    
    if args.train:
        # Train fusion model (placeholder)
        detector.train_fusion_model(None, None)
    else:
        # Perform detection
        if not args.image and not args.audio:
            print("❌ Please provide --image and/or --audio paths")
            return
            
        detector.detect_deepfake(args.image, args.audio)

if __name__ == "__main__":
    main() 
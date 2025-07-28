import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
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

# Configuration
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'faces')
REAL_FACES_PATH = os.path.join(PROCESSED_DATA_PATH, 'real')
FAKE_FACES_PATH = os.path.join(PROCESSED_DATA_PATH, 'fake')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'deepfake_detector_model.h5')
IMG_SIZE = 299 # Xception expects 299x299
BATCH_SIZE = 16
EPOCHS = 5

def load_dataset():
    data, labels = [], []
    for label, path in enumerate([REAL_FACES_PATH, FAKE_FACES_PATH]):
        for filename in os.listdir(path):
            if filename.endswith(('.png', '.jpg')):
                img_path = os.path.join(path, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    data.append(image)
                    labels.append(label)
    if not data: return None, None
    return np.array(data), np.array(labels)

def build_xception_model(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def predict_single_image(image_path, model):
    image = cv2.imread(image_path)
    if image is None: print("Error: Could not read image."); return
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    processed_image = preprocess_input(np.expand_dims(image, axis=0))
    prediction = model.predict(processed_image)[0][0]
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "DEEPFAKE" if prediction > 0.5 else "AUTHENTIC"
    print(f"Result: {label} | Confidence: {confidence:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Deepfake Image Detector (Xception).")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'])
    parser.add_argument('--image', type=str, help="Path to image for prediction.")
    args = parser.parse_args()

    if args.mode == 'predict':
        if not args.image: print("Error: --image path is required for predict mode.")
        elif not os.path.exists(MODEL_PATH): print(f"Error: Model not found at {MODEL_PATH}.")
        else:
            model = load_model(MODEL_PATH)
            predict_single_image(args.image, model)
    elif args.mode == 'train':
        data, labels = load_dataset()
        if data is not None:
            data = preprocess_input(data)
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
            model = build_xception_model((IMG_SIZE, IMG_SIZE, 3))
            model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)
            model.save(MODEL_PATH)
            print(f"✅ Advanced (Xception) model saved to {MODEL_PATH}")
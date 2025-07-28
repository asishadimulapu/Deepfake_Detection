# Deepfake Detection

A comprehensive deepfake detection system that can analyze audio, video, and images to identify synthetic media.

## 🚀 Features

- **🎙 Audio Detection**: Analyze audio files to detect synthetic speech.
- **🎥 Video Detection**: Detect deepfake videos using facial analysis and frame-level features.
- **🖼 Image Detection**: Identify manipulated or GAN-generated images.
- **🔗 Multimodal Fusion**: Combine audio, video, and image detectors for higher accuracy and robustness.

## 📁 Project Structure
```bash
Deepfake_agent/
├── deepfake_detection/
│ ├── audio_detect.py # Audio deepfake detection
│ ├── video_detect.py # Video deepfake detection
│ ├── image_detect.py # Image deepfake detection
│ └── multimodal_fusion.py # Multimodal detection fusion
├── models/ # Pre-trained models (not tracked in Git)
├── datasets/ # Raw datasets (not tracked in Git)
├── processed_data/ # Preprocessed data (not tracked in Git)
├── test_multimodal.py # Script to test the multimodal system
└── README.md # Project documentation
bash
Copy
Edit

## 🛠 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/asishadimulapu/Deepfake_Detection.git
cd Deepfake_Detection
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Make sure you’re using Python 3.8+ and have  opencv, and librosa installed.

🔍 Usage
🎧 Audio Detection
python
Copy
Edit
from deepfake_detection.audio_detect import AudioDetector

detector = AudioDetector()
result = detector.detect("path/to/audio.flac")
print("Audio Deepfake:", result)
🎬 Video Detection
python
Copy
Edit
from deepfake_detection.video_detect import VideoDetector

detector = VideoDetector()
result = detector.detect("path/to/video.mp4")
print("Video Deepfake:", result)
🖼 Image Detection
python
Copy
Edit
from deepfake_detection.image_detect import ImageDetector

detector = ImageDetector()
result = detector.detect("path/to/image.jpg")
print("Image Deepfake:", result)
🔗 Multimodal Detection
python
Copy
Edit
from deepfake_detection.multimodal_fusion import MultimodalDetector

detector = MultimodalDetector()
result = detector.detect_multimodal(
    audio_path="path/to/audio.flac",
    video_path="path/to/video.mp4"
)
print("Multimodal Decision:", result)
🧪 Testing
Run the multimodal test script:

bash
Copy
Edit
python test_multimodal.py
📦 Dataset
This project uses the LA Dataset (Logical Access subset of ASVspoof) for audio deepfakes and popular public deepfake datasets for video/image (Celeb-DF).
Due to size constraints, datasets are not included in this repository.

🤖 Models
Pre-trained model weights are stored in the models/ directory. These are large files and should be downloaded manually or handled via external storage.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributing
Fork this repository

Create a new feature branch:
git checkout -b feature/my-feature

Commit your changes:
git commit -m "Add new feature"

Push to the branch:
git push origin feature/my-feature

Open a Pull Request

🙏 Acknowledgments
ASVspoof LA dataset

Celeb-DF

OpenCV, TensorFlow, Keras, and Librosa

Researchers and contributors to deepfake detection models and open datasets

💡 “Detect truth in the age of deception.”

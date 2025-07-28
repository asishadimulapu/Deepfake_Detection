# Deepfake Detection System

<div align="center">

**“Detecting truth in the age of digital deception.”**

</div>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-green.svg" alt="Python Version"></a>
  <a href="#"><img src="https://img.shields.io/badge/Build-Passing-brightgreen" alt="Build Status"></a>
</p>

A comprehensive, multimodal deepfake detection system designed to analyze and identify synthetic media across audio, video, and image formats. This project leverages state-of-the-art deep learning models to provide robust and accurate detection capabilities.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Download Models & Datasets](#download-models--datasets)
- [Usage](#-usage)
  - [Audio Detection](#-audio-detection)
  - [Video Detection](#-video-detection)
  - [Image Detection](#-image-detection)
  - [Multimodal Detection](#-multimodal-detection)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Citation](#-citation)

## ✨ Features

- **🎙️ Audio Forensics**: Analyzes audio waveforms and spectrograms using models like RawNet2 to detect sophisticated text-to-speech (TTS) and voice conversion (VC) artifacts.
- **🎥 Video Analysis**: Employs deep learning models (e.g., EfficientNet, XceptionNet) to perform frame-level analysis, detecting visual inconsistencies, facial manipulation artifacts, and temporal irregularities.
- **🖼️ Image Scrutiny**: Identifies GAN-generated or synthetically altered images by analyzing subtle patterns, frequency domain artifacts, and compression signatures.
- **🔗 Multimodal Fusion**: Implements a fusion mechanism that synergistically combines evidence from audio and video streams, significantly improving detection accuracy and reducing false positives.

## 📁 System Architecture

The project is structured to separate concerns for each detection modality and facilitate easy extension.

```bash
Deepfake_Detection/
├── deepfake_detection/
│   ├── audio_detect.py        # Audio deepfake detection module
│   ├── video_detect.py        # Video deepfake detection module
│   ├── image_detect.py        # Image deepfake detection module
│   └── multimodal_fusion.py   # Multimodal detection and fusion logic
├── models/                      # Pre-trained model weights (download separately)
├── datasets/                    # Raw datasets (download separately)
├── processed_data/              # Preprocessed data for training/testing
├── test_multimodal.py           # End-to-end test script for the system
└── README.md                    # Project documentation
```

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.8 or higher
- Git
- `pip` for package management

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/asishadimulapu/Deepfake_Detection.git](https://github.com/asishadimulapu/Deepfake_Detection.git)
    cd Deepfake_Detection
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The `requirements.txt` file contains all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    This will install core libraries such as [TensorFlow](https://www.tensorflow.org/), [OpenCV](https://opencv.org/), and [Librosa](https://librosa.org/doc/latest/index.html).

### Download Models & Datasets

Pre-trained models and datasets are not tracked by Git due to their large size.

1.  **Models:** Download the pre-trained model weights from `[YOUR_DOWNLOAD_LINK_HERE]` and place them in the `models/` directory.

2.  **Datasets:** To replicate training or evaluation, download the following datasets and place them in the `datasets/` directory:
    * **Audio:** [ASVspoof 2019: Logical Access (LA)](https://datashare.ed.ac.uk/handle/10283/3336)
    * **Video/Image:** [Celeb-DF (v2)](https://github.com/yuezunli/celeb-deepfakeforensics)

## 🔍 Usage

The detection modules are designed to be simple and modular. Each detector returns a dictionary containing the prediction and a confidence score.

### 🎧 Audio Detection

```python
from deepfake_detection.audio_detect import AudioDetector

# Initialize the detector (loads the model)
audio_detector = AudioDetector()

# Perform detection
result = audio_detector.detect("path/to/your/audio.flac")
print(f"Audio Detection Result: {result}")
# Expected Output: {'prediction': 'fake', 'confidence': 0.92}
```

### 🎬 Video Detection

```python
from deepfake_detection.video_detect import VideoDetector

video_detector = VideoDetector()
result = video_detector.detect("path/to/your/video.mp4")
print(f"Video Detection Result: {result}")
# Expected Output: {'prediction': 'real', 'confidence': 0.85}
```

### 🖼️ Image Detection

```python
from deepfake_detection.image_detect import ImageDetector

image_detector = ImageDetector()
result = image_detector.detect("path/to/your/image.jpg")
print(f"Image Detection Result: {result}")
# Expected Output: {'prediction': 'fake', 'confidence': 0.98}
```

### 🔗 Multimodal Detection

The multimodal detector leverages both audio and video streams for a more robust final decision.

```python
from deepfake_detection.multimodal_fusion import MultimodalDetector

multimodal_detector = MultimodalDetector()
final_decision = multimodal_detector.detect_multimodal(
    video_path="path/to/your/video.mp4",
    audio_path="path/to/your/audio.flac" # Optional, can be extracted from video
)
print(f"Multimodal Fusion Decision: {final_decision}")
```

## 🧪 Testing

To verify the installation and the end-to-end functionality of the multimodal system, run the provided test script. Ensure you have sample media files available.

```bash
python test_multimodal.py
```

## 🗺️ Roadmap

We are continuously working to improve the system. Future plans include:
- [ ] **Real-time Detection:** Implement a high-performance pipeline for real-time video stream analysis.
- [ ] **Explainable AI (XAI):** Integrate methods like GRAD-CAM to visualize what the model is "looking at," providing interpretable results.
- [ ] **Expanded Model Zoo:** Add support for more state-of-the-art detection models.
- [ ] **API Service:** Package the detector into a REST API for easy integration into other applications.

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork this repository.
2.  Create a new feature branch (`git checkout -b feature/your-amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/your-amazing-feature`).
5.  Open a Pull Request.

Please ensure your code adheres to project standards and includes relevant tests.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgments

This work would not have been possible without the following resources:
* The organizers and dataset of the **[ASVspoof Challenge](https://www.asvspoof.org/)**.
* The creators of the **[Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)** dataset.
* The open-source communities behind **[OpenCV](https://opencv.org/)**, **[TensorFlow](https://www.tensorflow.org/)**, **[Keras](https://keras.io/)**, and **[Librosa](https://librosa.org/)**.
* All researchers and developers who have published their models and datasets for the advancement of deepfake detection.

## 📜 Citation

If you use this code in your research, please consider citing it:
```bibtex
@software{DeepfakeDetectionSystem2025,
  author = {Adimulapu, Asish},
  title = {{A Comprehensive Multimodal Deepfake Detection System}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/asishadimulapu/Deepfake_Detection](https://github.com/asishadimulapu/Deepfake_Detection)}}
}
```

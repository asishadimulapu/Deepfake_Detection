# Deepfake Detection

A comprehensive deepfake detection system that can analyze audio, video, and images to identify synthetic media.

## Features

- **Audio Detection**: Analyze audio files to detect synthetic speech
- **Video Detection**: Detect deepfake videos using facial analysis
- **Image Detection**: Identify manipulated images
- **Multimodal Fusion**: Combine multiple detection methods for improved accuracy

## Project Structure

```
Deepfake_agent/
├── deepfake_detection/
│   ├── audio_detect.py      # Audio deepfake detection
│   ├── video_detect.py      # Video deepfake detection
│   ├── image_detect.py      # Image deepfake detection
│   └── multimodal_fusion.py # Multimodal detection fusion
├── models/                  # Pre-trained models
├── datasets/               # Dataset files (not tracked in Git)
├── processed_data/         # Processed data (not tracked in Git)
└── test_multimodal.py     # Testing script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Deepfake_Detection.git
cd Deepfake_Detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Audio Detection
```python
from deepfake_detection.audio_detect import AudioDetector

detector = AudioDetector()
result = detector.detect("path/to/audio/file.flac")
```

### Video Detection
```python
from deepfake_detection.video_detect import VideoDetector

detector = VideoDetector()
result = detector.detect("path/to/video/file.mp4")
```

### Multimodal Detection
```python
from deepfake_detection.multimodal_fusion import MultimodalDetector

detector = MultimodalDetector()
result = detector.detect_multimodal(
    audio_path="path/to/audio.flac",
    video_path="path/to/video.mp4"
)
```

## Testing

Run the test script to evaluate the system:
```bash
python test_multimodal.py
```

## Dataset

The project uses the LA dataset for training and evaluation. Due to size constraints, the dataset files are not included in this repository.

## Models

Pre-trained models are stored in the `models/` directory. These files are large and not tracked in Git.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LA dataset for providing the training data
- OpenCV for computer vision capabilities
- TensorFlow/Keras for deep learning models 
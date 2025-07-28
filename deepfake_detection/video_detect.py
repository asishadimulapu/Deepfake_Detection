import os
import cv2 # OpenCV for video and image processing
import numpy as np

# --- Configuration ---
# Input paths (where your videos are)
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'videos')
REAL_VIDEO_PATH = os.path.join(DATASET_PATH, 'real')
FAKE_VIDEO_PATH = os.path.join(DATASET_PATH, 'fake')

# Output paths (where extracted faces will be saved)
# We create a new top-level folder for our processed data
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'faces')
OUTPUT_REAL_PATH = os.path.join(OUTPUT_PATH, 'real')
OUTPUT_FAKE_PATH = os.path.join(OUTPUT_PATH, 'fake')

# Face detection model files (you will need to download these)
# These files are part of OpenCV's deep neural network (DNN) samples.
# URL for prototxt: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
# URL for model: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
MODEL_PROTO_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'deploy.prototxt.txt')
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'res10_300x300_ssd_iter_140000.caffemodel')

# --- Helper Functions ---

def ensure_output_dirs_exist():
    """Create the output directories if they don't already exist."""
    print("Checking for output directories...")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"Created directory: {OUTPUT_PATH}")
    if not os.path.exists(OUTPUT_REAL_PATH):
        os.makedirs(OUTPUT_REAL_PATH)
        print(f"Created directory: {OUTPUT_REAL_PATH}")
    if not os.path.exists(OUTPUT_FAKE_PATH):
        os.makedirs(OUTPUT_FAKE_PATH)
        print(f"Created directory: {OUTPUT_FAKE_PATH}")
    print("✅ Output directories are ready.")

def load_face_detector():
    """Loads the pre-trained face detection model from disk."""
    print("Loading face detector model...")
    if not os.path.exists(MODEL_PROTO_PATH) or not os.path.exists(MODEL_WEIGHTS_PATH):
        print("❌ Error: Face detector model files not found.")
        print(f"Please download 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel'")
        print(f"And place them in a 'models' folder in your project root.")
        return None
    
    net = cv2.dnn.readNetFromCaffe(MODEL_PROTO_PATH, MODEL_WEIGHTS_PATH)
    print("✅ Face detector loaded successfully.")
    return net

def process_video(video_path, output_folder, face_detector, frame_interval=15):
    """
    Extracts faces from a single video file and saves them as images.
    
    Args:
        video_path (str): The path to the input video.
        output_folder (str): The directory to save the cropped face images.
        face_detector: The loaded OpenCV DNN model.
        frame_interval (int): Process every Nth frame to speed things up.
    """
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"\nProcessing video: {video_name}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_index = 0
    faces_saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Only process every Nth frame
        if frame_index % frame_interval == 0:
            (h, w) = frame.shape[:2]
            # Create a blob from the image and pass it through the network
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            detections = face_detector.forward()

            # Find the detection with the highest confidence
            best_detection_index = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, best_detection_index, 2]

            # Ensure the detection confidence is above a threshold (e.g., 50%)
            if confidence > 0.5:
                # Get the coordinates of the bounding box
                box = detections[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box is within the frame dimensions
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # Extract the face ROI (Region of Interest)
                face = frame[startY:endY, startX:endX]

                # Ensure the extracted face is not empty
                if face.size != 0:
                    # Save the cropped face to the output directory
                    output_filename = f"{video_name}_frame{frame_index}_face.png"
                    save_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(save_path, face)
                    faces_saved += 1
        
        frame_index += 1
    
    print(f"-> Finished. Saved {faces_saved} faces from {video_name}.")
    cap.release()


if __name__ == '__main__':
    ensure_output_dirs_exist()
    net = load_face_detector()

    if net:
        # --- Process REAL videos ---
        print("\n--- Scanning REAL videos ---")
        real_videos = [os.path.join(REAL_VIDEO_PATH, v) for v in os.listdir(REAL_VIDEO_PATH) if v.endswith('.mp4')]
        # Process a small subset to start (e.g., the first 5)
        for video_file in real_videos[:5]:
            process_video(video_file, OUTPUT_REAL_PATH, net)

        # --- Process FAKE videos ---
        print("\n--- Scanning FAKE videos ---")
        fake_videos = [os.path.join(FAKE_VIDEO_PATH, v) for v in os.listdir(FAKE_VIDEO_PATH) if v.endswith('.mp4')]
        # Process a small subset to start (e.g., the first 5)
        for video_file in fake_videos[:5]:
            process_video(video_file, OUTPUT_FAKE_PATH, net)
            
        print("\n\n✅ All processing complete!")
        print(f"Check the '{OUTPUT_REAL_PATH}' and '{OUTPUT_FAKE_PATH}' folders for the extracted faces.")


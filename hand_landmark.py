import cv2
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Function to extract frames from a video
def extract_frames(video_path, output_folder):
    """
    Extracts all frames from a video and saves them as images in the specified folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where the extracted frames will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_filename = os.path.join(output_folder, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")
        frame_idx += 1

    cap.release()
    print(f"Extraction complete. Total frames saved: {frame_idx}")
    return frame_idx  # Return the total number of frames

# Function to detect hands and generate bounding boxes
def detect_hands_and_generate_bboxes(image_path, detector):
    """
    Detects hands in an image and generates bounding boxes.

    Args:
        image_path (str): Path to the input image file.
        detector: MediaPipe Hand Landmarker object.

    Returns:
        List of bounding boxes for the detected hands.
    """
    # Load the image
    image = mp.Image.create_from_file(image_path)

    # Detect hand landmarks
    detection_result = detector.detect(image)

    # Extract bounding boxes from the landmarks
    bounding_boxes = []
    for hand_landmarks in detection_result.hand_landmarks:
        x_coords = [landmark.x * image.width for landmark in hand_landmarks]
        y_coords = [landmark.y * image.height for landmark in hand_landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        bounding_boxes.append([x_min, y_min, x_max, y_max])

    return bounding_boxes

# Main function to process the video
def process_video(video_path, output_folder, bbox_json_path):
    """
    Processes a video to extract frames, detect hands, and save bounding boxes.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where the extracted frames will be saved.
        bbox_json_path (str): Path to the JSON file where bounding boxes will be saved.
    """
    # Step 1: Extract frames from the video
    total_frames = extract_frames(video_path, output_folder)

    # Step 2: Initialize MediaPipe Hand Landmarker
    model_path = 'hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Step 3: Process each frame and generate bounding boxes
    bbox_data = {}
    for frame_idx in range(total_frames):
        frame_filename = os.path.join(output_folder, f"{frame_idx:05d}.jpg")
        bounding_boxes = detect_hands_and_generate_bboxes(frame_filename, detector)
        bbox_data[frame_idx] = bounding_boxes
        print(f"Processed frame {frame_idx}: {bounding_boxes}")

    # Step 4: Save bounding boxes to a JSON file
    with open(bbox_json_path, 'w') as f:
        json.dump(bbox_data, f, indent=4)
    print(f"Bounding boxes saved to {bbox_json_path}")

# Run the pipeline
video_path = "test.mp4"
output_folder = "hands/"
bbox_json_path = "bounding_boxes.json"
process_video(video_path, output_folder, bbox_json_path)
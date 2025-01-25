import cv2
import os

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

video_path = "test.mp4"
output_folder = "hands/"
extract_frames(video_path, output_folder)
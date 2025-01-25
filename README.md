# **Automatic Hand Tracking Pipeline**

This project implements an automatic pipeline for tracking hand movements in a video using OpenCV, MediaPipe, and SAM 2. It includes frame extraction, hand detection, and mask generation for tracking hands across video frames.

## **Features**
- **Frame Extraction:** Extracts frames from the input video for processing.
- **Hand Detection:** Uses MediaPipe Hand Landmarker to detect hands and generate bounding boxes in each frame.
- **Hand Masking:** Leverages SAM 2 to generate masks for hands in each frame based on bounding boxes.
- **Output Video Generation:** Annotates the video with hand masks and saves the result.

---

## **Setup**

### **1. Clone the Repository**
```bash
git clone https://github.com/mannadamay12/sam2-ad.git
cd sam2-ad
```

### **2. Environment Setup**
1. **Create and Activate Virtual Environment:**
   ```bash
   conda create -n sam2 python=3.12
   conda activate sam2
   ```
2. **Install Dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install mediapipe opencv-python
   ```

3. **Set Up SAM 2:**
   Follow [SAM 2 Setup Instructions](https://github.com/facebookresearch/sam2).

### **3. Project Directory Structure**
```plaintext
mannadamay12-sam2-ad/
├── bounding_boxes.json         # JSON file for bounding box data
├── extract_frames.py           # Frame extraction script
├── hand_landmark.py            # Hand detection and bounding box generation
├── mediapipe_video.py          # Annotated video generation
├── sam2.ipynb                  # SAM 2 integration for hand masking
└── hands/                      # Directory for extracted frames
```

---

## **Usage**

### **1. Extract Frames**
Extract frames from the input video for processing:
```bash
python extract_frames.py
```

### **2. Hand Detection and Bounding Box Generation**
Detect hands and generate bounding boxes:
```bash
python hand_landmark.py
```

### **3. Generate Annotated Video with Hand Masks**
Use SAM 2 to create hand masks and generate the output video:
```bash
python mediapipe_video.py
```

---

## **Deliverables**
1. **Code Repository:** [GitHub](https://github.com/mannadamay12/hand-tracking-pipeline) (includes organized code, README, and pip requirements).
2. **Output Video Demo:** Annotated video with masked hands.

---

## **References**
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [SAM 2 Repository](https://github.com/facebookresearch/sam2)
- [SAM 2 Video Predictor Example](https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

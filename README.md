# Real-Time Face Analysis

This project uses the `DeepFace` library and OpenCV to perform real-time analysis of facial attributes, including age, gender, and emotions, from a live camera feed.

## Features
- Detects faces in real-time using a connected camera.
- Analyzes facial attributes:
  - **Age**
  - **Gender**
  - **Emotion**
- Displays the analyzed attributes directly on the video feed.

## Prerequisites
### Python Libraries
Ensure that all dependencies listed in the `requirements.txt` file are installed.

You can install them using pip:
```bash
pip install -r requirements.txt
```

### Hardware Requirements
- A camera device connected to your computer.
- A system capable of running Python and TensorFlow efficiently.

## How to Use

1. **Clone the Repository or Save the Script**
   Save the provided script into a `.py` file, for example, `real_time_face_analysis.py`.

2. **Run the Script**
   Execute the script using Python:
   ```bash
   python real_time_face_analysis.py
   ```

3. **Interact with the Program**
   - The camera feed will open in a new window.
   - Press `q` to exit the program.

## Code Overview

### Importing Necessary Libraries
```python
import os
import cv2
from deepface import DeepFace
```
These libraries are essential for accessing the camera, performing face detection, and analyzing attributes.

### Disable TensorFlow oneDNN Optimizations
```python
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```
This ensures numerical consistency and avoids unnecessary warnings.

### Face Analysis Function
The `analyze_faces` function uses DeepFace to analyze the input video frame for facial attributes and overlays the results on the frame.

### Main Loop
- Opens the camera feed using OpenCV.
- Continuously captures frames, processes them, and displays the analyzed video.
- Press `q` to terminate the program.

## Troubleshooting
- **Error: Could not open camera**: Ensure your camera is connected and accessible.
- **Low performance**: Run on a system with a GPU for faster analysis.
- **Missing dependencies**: Ensure all required Python libraries are installed.

## Acknowledgments
- [DeepFace](https://github.com/serengil/deepface): For real-time face analysis.
- [OpenCV](https://opencv.org/): For accessing and processing video feeds.

## License
This project is licensed under the MIT License.

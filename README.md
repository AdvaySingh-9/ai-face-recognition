# Face Recognition System

A Python-based real-time face recognition system that uses OpenCV and LBPH (Local Binary Patterns Histograms) face recognizer. Capture training images, train a machine learning model, and recognize faces in real-time from your webcam.

## Features

- **Face Capture**: Capture training images of multiple people with automatic face detection
- **Model Training**: Train an LBPH face recognizer model using the captured dataset
- **Real-time Recognition**: Recognize faces in real-time using your webcam
- **Confidence Scoring**: Display confidence values for recognized faces
- **Easy-to-use CLI**: Interactive command-line menu for all operations
- **Persistent Models**: Save trained models and label mappings for later use

## Requirements

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy
- Webcam/Camera device

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AdvaySingh-9/ai-face-recognition

```

### 2. Install Dependencies

```bash
pip install opencv-contrib-python numpy
```

**Note**: `opencv-contrib-python` is required for the LBPH Face Recognizer module.

### 3. Verify Installation

```bash
python -c "import cv2; print(cv2.__version__)"
```

## Usage

### Run the Program

```bash
python face_recognitation.py
```

### Menu Options

#### 1. Capture Faces

- Select option `1` from the menu
- Enter the name of the person
- Look at the webcam and allow the program to capture face images
- Press `q` to stop capturing
- Images are saved in the `dataset/` directory

#### 2. Train Model

- Select option `2` from the menu
- The program will:
  - Load all training images from the `dataset/` directory
  - Train the LBPH face recognizer
  - Save the trained model as `face_recognizer.yml`
  - Save the label mapping as `label_map.pkl`

#### 3. Recognize Faces

- Select option `3` from the menu
- Ensure the model has been trained (option 2)
- The program will use your webcam to detect and recognize faces
- Recognized faces are labeled with their name and confidence score
- Unrecognized faces are labeled as "Animal: Crow"
- Press `q` to stop recognition

#### 4. Exit

- Select option `4` to exit the program

## Project Structure

```
ai-face-recognition/
├── face_recognitation.py       # Main program
├── dataset/                    # Training dataset directory (auto-created)
│   ├── person1/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └── person2/
│       ├── 0.jpg
│       └── ...
├── face_recognizer.yml         # Trained model (auto-created)
├── label_map.pkl               # Label mapping (auto-created)
└── README.md                   # This file
```

## How It Works

### 1. Face Detection

The system uses **Haar Cascade Classifiers** to detect faces in video frames. This is a pre-trained model that comes with OpenCV.

### 2. Model Training

The **LBPH (Local Binary Patterns Histograms)** face recognizer analyzes the texture patterns in face images to create a model that can distinguish between different people.

### 3. Face Recognition

When recognizing, the system:

- Detects faces in the webcam feed
- Compares them against the trained model
- Returns a confidence score (lower = more confident)
- Displays the name if confidence < 70

## Configuration

You can modify these constants in `face_recognitation.py`:

```python
FACE_SIZE = (200, 200)  # Size of face images for training/recognition
```

Adjust this for different face image resolutions.

## Parameters

- **detectMultiScale parameters** (in `recognize_faces` function):
  - `1.3` - Scale factor (how much image size is reduced at each scale)
  - `5` - Minimum neighbors (how many neighbors should be detected for final classification)
  - `minSize=(50, 50)` - Minimum face size to detect

## Troubleshooting

| Issue                          | Solution                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------- |
| "No training data found"       | Make sure you've captured faces using option 1                                   |
| "Please train the model first" | Use option 2 to train the model before recognition                               |
| Camera not working             | Check if your webcam is connected and accessible                                 |
| Low recognition accuracy       | Capture more images (increase the limit from 100) or improve lighting conditions |
| Face not detected              | Ensure sufficient lighting and that your face is clearly visible from front      |

## Tips for Better Results

1. **Lighting**: Use good, even lighting when capturing training images
2. **Variety**: Capture faces from different angles and distances
3. **Quantity**: Capture 100+ images per person for better accuracy
4. **Background**: Use a simple, consistent background during training
5. **Camera**: Ensure the camera is stable and in focus

## Limitations

- Works best with frontal face images
- Performance depends on image quality and lighting
- May have difficulty with sunglasses, hats, or other face coverings
- Requires retraining if adding new people to the system

## Future Enhancements

- [ ] Support for multiple face recognition algorithms (CNN, DNN)
- [ ] Batch processing for multiple faces
- [ ] Database integration for storing recognition history
- [ ] WebUI/GUI interface
- [ ] Support for age and gender estimation
- [ ] Real-time performance metrics

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## Author

This project is created by a solo developer: [Advay Singh](https://advay-portfolio.netlify.app)

## Disclaimer

This project is for educational purposes. Use responsibly and ensure compliance with local privacy and surveillance laws when deploying face recognition systems.

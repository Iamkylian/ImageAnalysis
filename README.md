# Anime Face Detection with YOLOv8

This project uses a YOLOv8 model trained on anime faces to detect and analyze anime characters in images. It processes images, detects anime faces, and outputs the results in both visual and data formats.

## Prerequisites

- Python 3.7+
- PyTorch
- Ultralytics
- pandas
- PyYAML

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Iamkylian/ImageAnalysis.git
2. Install the required packages:
   ```bash
   pip install torch ultralytics pandas pyyaml
3. Download the YOLOv8 Anime model:
* Download the model.pt file from this link
* Place the model.pt file in the models/anime_models/ directory of the project

## Usage
1. Place your anime images in the data/images/ directory.
2. Run the script:
    ```bash
    python main.py
    ```
3. The script will process all images in the data/images/ directory and save the results in the output/ directory. For each image, it will create:
* A new image with bounding boxes around detected faces
* A CSV file with detection data (coordinates, confidence, class)

## Output
For each processed image, you'll find:
* {image_name}_detections.jpg: The original image with bounding boxes drawn around detected faces
* {image_name}_detections.csv: A CSV file containing detection data (coordinates, confidence, class)
import yaml
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import os

# Load configuration from config.yaml


def load_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load the YOLO Anime model (YOLOv8)


def load_yolo_anime_model(model_path):
    try:
        # Load the YOLOv8 model from the specified path
        model = YOLO(model_path)
        print("YOLOv8 Anime model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the YOLO Anime model: {e}")
        return None

# Detect faces in an anime image


def detect_anime_faces(image_path, model, output_dir):
    try:
        # Load the image
        img = Image.open(image_path)

        # Perform detection with the YOLO Anime model
        results = model(img)  # The model returns a list of results
        print("Model results:")
        print(results)

        # Check results
        if results:
            print("Detection successful.")

            # Check detection boxes
            detections = results[0].boxes  # Get detection objects
            if detections is not None and detections.xyxy.shape[0] > 0:
                # Create a list to store results
                detection_list = []

                # Iterate over detections and extract information
                for i in range(detections.xyxy.shape[0]):
                    # Convert coordinates to numpy array
                    box = detections.xyxy[i].cpu().numpy()
                    # Retrieve confidence as float
                    conf = detections.conf[i].cpu().item()
                    # Retrieve class as float
                    cls = detections.cls[i].cpu().item()

                    # Add the information to the list
                    detection_list.append({
                        "xmin": box[0],
                        "ymin": box[1],
                        "xmax": box[2],
                        "ymax": box[3],
                        "confidence": conf,
                        "class": cls
                    })

                # Convert the list to a DataFrame
                detections_df = pd.DataFrame(detection_list)

                # Display the image with detections
                img_with_boxes = results[0].plot()

                # Define the output path for the image
                img_name = os.path.basename(image_path).split('.')[0]
                img_output_path = os.path.join(
                    output_dir, f'{img_name}_detections.jpg')
                Image.fromarray(img_with_boxes).save(img_output_path)
                print(
                    f"The image with detections has been saved as: {img_output_path}")

                # Save the results in a CSV file
                csv_output_path = os.path.join(
                    output_dir, f'{img_name}_detections.csv')
                detections_df.to_csv(csv_output_path, index=False)
                print(
                    f"Detection results have been saved in {csv_output_path}")

                return detections_df
            else:
                print("No detection boxes found.")
                return pd.DataFrame()  # Return an empty DataFrame if no detections are found
        else:
            print("No detections found in the results.")
            return pd.DataFrame()  # Return an empty DataFrame if no detections are found
    except Exception as e:
        print(f"Error during detection on the image: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there is an error

# Main function


def main(input_dir):
    # Load the configuration and the model
    config = load_config()
    yolo_anime_model_path = config['models']['anime_yolo']
    yolo_anime_model = load_yolo_anime_model(yolo_anime_model_path)

    if yolo_anime_model:
        # Process all image files in the input directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check image formats
                image_path = os.path.join(input_dir, filename)

                # Create an output directory for each image
                output_dir = os.path.join(
                    config['output']['save_path'], os.path.splitext(filename)[0])
                os.makedirs(output_dir, exist_ok=True)

                # Execute detection on the specified image
                detections = detect_anime_faces(
                    image_path, yolo_anime_model, output_dir)
                print("Displaying detections:")
                print(detections)

                # Print results as a table
                if not detections.empty:
                    print("Anime face detections:")
                    print(detections)
                else:
                    print("No detections found.")
    else:
        print("The model could not be loaded.")


# Entry point of the script
if __name__ == "__main__":
    input_directory = 'data/images/'  # Path to the directory containing the images
    main(input_directory)

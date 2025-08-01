# export_models.py
from ultralytics import YOLO
import logging

# Configure logging to see progress
logging.basicConfig(level=logging.INFO)


def main():
    """
    Automatically downloads the official yolo11n and yolo11s models,
    then exports them to the OpenVINO FP32 format.
    """
    models_to_export = ['yolo11n.pt', 'yolo11s.pt']

    for model_name in models_to_export:
        logging.info(f"--- Processing {model_name} ---")

        # The YOLO() constructor will automatically download the model if it doesn't exist
        logging.info(f"Loading/Downloading {model_name}...")
        model = YOLO(model_name)

        # Export the model to OpenVINO FP32 format
        logging.info(f"Exporting {model_name} to OpenVINO format...")
        model.export(format="openvino", device="cpu")

        logging.info(f"Export finished for {model_name}.\n")


if __name__ == "__main__":
    main()

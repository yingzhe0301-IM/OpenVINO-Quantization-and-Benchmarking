# download_dataset.py
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import logging

# Configure logging to see the process
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def main():
    """
    Downloads 300 sample images from the COCO-2017 validation set
    and saves them to a local directory for use in quantization.
    """
    export_dir = Path("coco-2017-images")

    # Check if the directory already exists to avoid re-downloading
    if export_dir.exists() and any(export_dir.iterdir()):
        log.info(f"Dataset already exists in '{export_dir}'. Skipping download.")
    else:
        log.info("Downloading COCO-2017 validation split (300 samples)...")
        # Load the dataset from the FiftyOne Zoo
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            max_samples=300,
            shuffle=True  # Shuffle to get a random subset
        )

        # Save the images to disk
        log.info(f"Exporting images to '{export_dir}'...")
        dataset.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.ImageDirectory,
        )
        log.info("Export complete.")


if __name__ == "__main__":
    main()

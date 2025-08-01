# quantize_models.py
from pathlib import Path
import click
import nncf
import openvino as ov
import torch
from torchvision import datasets, transforms
import logging

# Configure logging to see the process
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class ImagesOnlyFolder(torch.utils.data.Dataset):
    """A custom dataset class to load images from a folder."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_path = [f for f in Path(root).rglob("*.jpg") if f.is_file()]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image


@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True,
              help="Path to the FP32 OpenVINO model (.xml) to be quantized.")
@click.option("--dataset_path", type=click.Path(exists=True), required=True,
              help="Path to the calibration image folder.")
@click.option("--output_path", type=click.Path(), required=True, help="Path to save the quantized INT8 model (.xml).")
def nncf_ov_quantize(model_path: str, dataset_path: str, output_path: str):
    """Quantizes an OpenVINO model from FP32 to INT8 using NNCF."""
    log.info(f"--- Starting Quantization for {Path(model_path).name} ---")

    # Load the unquantized model
    model = ov.Core().read_model(model_path)

    # Get model's expected input size
    input_layer = model.input(0)
    height, width = input_layer.shape[2], input_layer.shape[3]
    log.info(f"Model expects input size: {height}x{width}")

    # Define the correct image transformations for YOLO
    prep_transform = [transforms.Resize((height, width)), transforms.ToTensor()]

    # Create the calibration dataset
    val_dataset = ImagesOnlyFolder(dataset_path, transform=transforms.Compose(prep_transform))
    dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    calibration_dataset = nncf.Dataset(dataset_loader)

    # Run the quantization process
    log.info("Running NNCF quantization... (This may take a few minutes)")
    quantized_model = nncf.quantize(model, calibration_dataset)

    # Save the quantized model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving quantized model to {output_path}")
    ov.save_model(quantized_model, str(output_path))
    log.info("--- Quantization Finished ---\n")


if __name__ == "__main__":
    nncf_ov_quantize()

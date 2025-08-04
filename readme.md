# **OpenVINO Quantization and Benchmarking for Object Detection Models on Ubuntu 24.04**

## 1. Project Goal

This project delivers a fully automated and reproducible pipeline for optimizing YOLO models with OpenVINO on Ubuntu 24.04. It guides users through the complete process—from model acquisition and conversion to INT8 quantization and performance benchmarking—targeting both CPU and integrated GPU (iGPU) inference.

Although configuring Intel iGPU acceleration on Ubuntu is often challenging and error-prone, this workflow provides a reliable, step-by-step guide that ensures the environment is properly set up—allowing users to fully enable iGPU inference simply by following the documented steps.

## 2. Required Scripts

This project contains 4 core scripts. Please ensure they are all in the same directory:
 • `download_dataset.py`
 • `export_models.py`
 • `quantize_models.py`
 • `run_and_parse_benchmarks.py`

## 3. Setup (One-Time Only)

The following steps will create the correct software environment and install the necessary system drivers. This process only needs to be done once.

Step 1: Create the Conda Environment
This creates an isolated environment with the correct Python version.

```bash
 conda create --name openvino_env python=3.12 -y
 conda activate openvino_env
```

Step 2: Install System Drivers

These commands install the correct Intel GPU compute drivers for Ubuntu 24.04.

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt update
sudo apt install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-media-va-driver-non-free libmfx-gen1 libvpl2 libva-glx2 va-driver-all
```

Step 3: Install OpenVINO via Conda

This is a critical step that installs the complete OpenVINO runtime, including the GPU plugin.

```bash
conda install -c conda-forge openvino=2025.2.0 -y
```

Step 4: Install Other Python Packages

```bash
pip install ultralytics nncf click fiftyone opencv-python rich
```

Step 5: Create Symbolic Link

This crucial step connects the Conda environment to the system's GPU driver.

```bash
mkdir -p ~/miniconda3/envs/openvino_env/etc/OpenCL/vendors
ln -s /etc/OpenCL/vendors/intel.icd ~/miniconda3/envs/openvino_env/etc/OpenCL/vendors/intel.icd
```

Step 6: Set Permissions and Reboot

This grants your user account permission to access the GPU hardware.

```bash
sudo gpasswd -a ${USER} render
sudo reboot
```

## 4. Execution Workflow

After completing the one-time setup and rebooting, run the following commands in order from your project directory. Make sure you have activated the correct conda environment (`conda activate openvino_env`).

Step 1: Download the Calibration Dataset

This script downloads 300 images from the COCO dataset, which are needed for the quantization step.

```bash
python download_dataset.py
```

Step 2: Download and Convert Models to OpenVINO FP32

This script automatically downloads yolo11n.pt and yolo11s.pt and then converts them to the standard FP32 OpenVINO format.

```bash
python export_models.py
```

Step 3: Quantize Models to INT8

These commands use the downloaded images to convert the FP32 models into the optimized INT8 format.

```bash
# Quantize the 'yolo11n' model
python quantize_models.py --model_path yolo11n_openvino_model/yolo11n.xml --dataset_path coco-2017-images --output_path yolo11n_openvino_model_int8/yolo11n.xml

# Quantize the 'yolo11s' model
python quantize_models.py --model_path yolo11s_openvino_model/yolo11s.xml --dataset_path coco-2017-images --output_path yolo11s_openvino_model_int8/yolo11s.xml
```

Step 4: Run All Benchmarks and Display Results

This Python script executes the official benchmark_app tool for all 8 model variations, captures the results, and prints a clean, formatted summary table.

```bash
python run_and_parse_benchmarks.py
```

## 5. Example Benchmark Results

After running the benchmark script, you should see something like this:

| Model   | Device | Precision | Throughput (FPS) |
|---------|--------|-----------|------------------|
| yolo11n | GPU.0  | FP32      | 65.52            |
| yolo11n | GPU.0  | INT8      | 89.37            |
| yolo11s | GPU.0  | FP32      | 27.15            |
| yolo11s | GPU.0  | INT8      | 42.67            |
| yolo11n | CPU    | FP32      | 20.76            |
| yolo11n | CPU    | INT8      | 40.07            |
| yolo11s | CPU    | FP32      |  6.90            |
| yolo11s | CPU    | INT8      | 16.23            |
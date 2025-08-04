# OpenVINO Quantization and Benchmarking for Object Detection Models on Ubuntu 24.04

---

## 🚀 Project Goal

This project delivers a fully automated and reproducible pipeline for optimizing YOLO models with OpenVINO on Ubuntu 24.04. It guides users through the complete process—from model acquisition and conversion to INT8 quantization and performance benchmarking—targeting both CPU and integrated GPU (iGPU) inference.

Although configuring Intel iGPU acceleration on Ubuntu can be challenging, this workflow provides a reliable, step-by-step guide that ensures the environment is properly set up—allowing users to fully enable iGPU inference simply by following the documented steps.

---

## 🔧 Prerequisites

Before you begin, make sure you have:

- **Operating System**: Ubuntu 24.04 LTS  
- **Conda**: version 4.x or newer (path examples assume Miniconda)
- **GPU drivers**: Intel GPU drivers installed (see Setup below)  
- **Network**: Internet access for downloading models and datasets  

---

## 📚 Required Scripts

All necessary scripts for this workflow are included in the repository:

- `download_dataset.py` —— Download 300 COCO images for quantization calibration  
- `export_models.py` —— Download YOLOv11n/s weights and convert to OpenVINO FP32  
- `quantize_models.py` —— Run INT8 quantization on FP32 OpenVINO models  
- `run_and_parse_benchmarks.py` —— Execute `benchmark_app` and generate a summary table  

---

## 🛠️ Setup

These steps install drivers and libraries—you only need to do this once.

1. **Create the Conda Environment**  
   ```bash
   conda create --name openvino_env python=3.12 -y
   conda activate openvino_env
   ```

2. **Install Intel GPU Compute Drivers**  
   ```bash
   sudo apt update
   sudo apt install -y software-properties-common
   sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
   sudo apt update
   sudo apt install -y \
     libze-intel-gpu1 libze1 intel-metrics-discovery \
     intel-opencl-icd clinfo intel-media-va-driver-non-free \
     libmfx-gen1 libvpl2 libva-glx2 va-driver-all
   ```

3. **Install OpenVINO via Conda**  
   ```bash
   conda install -c conda-forge openvino=2025.2.0 -y
   ```

4. **Install Python Packages**  
   ```bash
   pip install ultralytics click fiftyone opencv-python rich
   ```

5. **Link OpenCL ICD for Your Environment**  
   ```bash
   mkdir -p ~/miniconda3/envs/openvino_env/etc/OpenCL/vendors
   ln -s /etc/OpenCL/vendors/intel.icd \
       ~/miniconda3/envs/openvino_env/etc/OpenCL/vendors/intel.icd
   ```

6. **Grant GPU Access and Reboot**  
   ```bash
   sudo gpasswd -a ${USER} render
   sudo reboot
   ```

---

## ⚡️ Execution Workflow

After rebooting and re-activating the Conda environment (`conda activate openvino_env`), run each step from your project directory:

1. **Download the Calibration Dataset**  
   ```bash
   python download_dataset.py
   ```

2. **Export Models to OpenVINO FP32**  
   ```bash
   python export_models.py
   ```

3. **Quantize to INT8**  
   ```bash
   # Quantize yolo11n
   python quantize_models.py \
     --model_path yolo11n_openvino_model/yolo11n.xml \
     --dataset_path coco-2017-images \
     --output_path yolo11n_openvino_model_int8/yolo11n.xml

   # Quantize yolo11s
   python quantize_models.py \
     --model_path yolo11s_openvino_model/yolo11s.xml \
     --dataset_path coco-2017-images \
     --output_path yolo11s_openvino_model_int8/yolo11s.xml
   ```

4. **Run All Benchmarks**  
   ```bash
   python run_and_parse_benchmarks.py
   ```

---

## 📊 Example Benchmark Results

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

> **Note:** Actual numbers may vary depending on driver and hardware versions.

---

## 🔗 References

- [OpenVINO Documentation](https://docs.openvino.ai/)  
- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)  
- [OpenVINO Issue #28892](https://github.com/openvinotoolkit/openvino/issues/28892)  
- [RF-DETR Meets OpenVINO: Real-time INT8 Object Detection on an Intel iGPU](https://medium.com/latinxinai/rf-detr-meets-openvino-real-time-int8-object-detection-on-an-intel-igpu-da8ddba3de01)  

# run_and_parse_benchmarks.py
import subprocess
import re
from pathlib import Path
from rich.console import Console
from rich.table import Table


def run_single_benchmark(model_path: Path, device: str):
    """
    Runs a single benchmark_app command, captures and parses the output for FPS.
    """
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return None

    command = [
        "benchmark_app",
        "-m", str(model_path),
        "-d", device,
        "-niter", "300"
    ]

    # Run the command and capture output
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # Check for errors
    if result.returncode != 0:
        print(f"Error running benchmark for {model_path} on {device}.")
        # Print the last 500 characters of stderr for diagnosis
        print(result.stderr[-500:])
        return None

    # Parse the output to find the throughput
    output = result.stdout
    throughput_match = re.search(r"Throughput:\s+([\d\.]+)\s+FPS", output)

    if throughput_match:
        return float(throughput_match.group(1))
    else:
        print(f"Could not find throughput value for {model_path} on {device}.")
        return None


def main():
    """
    Defines and runs all 8 benchmark tests, then prints the results in a table.
    """
    benchmarks = [
        {"name": "yolo11n FP32", "path": Path("yolo11n_openvino_model/yolo11n.xml"), "device": "GPU.0"},
        {"name": "yolo11n INT8", "path": Path("yolo11n_openvino_model_int8/yolo11n.xml"), "device": "GPU.0"},
        {"name": "yolo11s FP32", "path": Path("yolo11s_openvino_model/yolo11s.xml"), "device": "GPU.0"},
        {"name": "yolo11s INT8", "path": Path("yolo11s_openvino_model_int8/yolo11s.xml"), "device": "GPU.0"},
        {"name": "yolo11n FP32", "path": Path("yolo11n_openvino_model/yolo11n.xml"), "device": "CPU"},
        {"name": "yolo11n INT8", "path": Path("yolo11n_openvino_model_int8/yolo11n.xml"), "device": "CPU"},
        {"name": "yolo11s FP32", "path": Path("yolo11s_openvino_model/yolo11s.xml"), "device": "CPU"},
        {"name": "yolo11s INT8", "path": Path("yolo11s_openvino_model_int8/yolo11s.xml"), "device": "CPU"},
    ]

    results = []
    for bench in benchmarks:
        print(f"--- Benchmarking {bench['name']} on {bench['device']} ---")
        fps = run_single_benchmark(bench['path'], bench['device'])
        results.append({"name": bench['name'], "device": bench['device'], "fps": fps})

    # Create and print the results table
    table = Table(title="yolo11 OpenVINO Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Device", style="yellow")
    table.add_column("Precision", style="magenta")
    table.add_column("Throughput (FPS)", style="green", justify="right")

    for res in results:
        # Correctly split "yolo11n" or "yolo11s" from "FP32" or "INT8"
        parts = res['name'].rsplit(' ', 1)
        if len(parts) == 2:
            model_name, precision = parts
        else:
            model_name, precision = res['name'], ""  # Fallback

        fps_str = f"{res['fps']:.2f}" if res['fps'] is not None else "Failed"
        table.add_row(model_name, res['device'], precision, fps_str)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()

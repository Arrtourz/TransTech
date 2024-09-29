
# Traffic Volume Analysis Tool

## Overview

The Traffic Volume Analysis Tool is a sophisticated computer vision application designed to analyze and quantify vehicle traffic in video footage. Utilizing state-of-the-art object detection and tracking algorithms, this tool provides accurate traffic volume measurements for urban planning, traffic management, and transportation research.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.4 or compatible version
- Python 3.10 or later

## Installation

1. Create a conda environment and activate it:

    ```bash
    conda create -n traffic_analysis python=3.10
    conda activate traffic_analysis
    ```

2. Install PyTorch, Torchvision, and Torchaudio with CUDA support:

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

    > **Note:** Ensure that you install the correct version of PyTorch and Torchvision compatible with your CUDA version.

3. Install the additional dependencies from the `requirements.txt` file:

    ```bash
    conda install -r requirements.txt
    ```

## Usage

### Define Detection Areas

Use the interactive interface to define the detection areas. Once done, press 'q' to save and exit.

```bash
python adjust_detection_area_box.py path/to/your/video.mp4
```

```bash
python adjust_detection_area_line.py path/to/your/video.mp4
```

### Traffic Volume Detection

#### Box Detection Mode:

```bash
python volumn_calculation.py path/to/your/video.mp4 box --box_file area_box.json
```

#### Line Detection Mode:

```bash
python volumn_calculation.py path/to/your/video.mp4 lines --lines_file area_lines.json
```

### Traffic Density Detection

Press 'q' to save and exit.

```bash
python density_calculation.py path/to/your/video.mp4
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

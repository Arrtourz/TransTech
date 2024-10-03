# Traffic Light Detection Tools

This repository contains a set of tools for detecting traffic lights in images and videos. Below are the available scripts and their usage.

## Scripts

### 1. Detect Traffic Lights in an Image
Use the following command to detect traffic lights in a sample image:

```bash
python detect_img.py sample\sample.jpg
```

### 2. Detect Traffic Lights in a Simple Video
Use the following command to detect traffic lights in a sample video:

```bash
python detect_video_simple.py sample\sample.mp4
```

### 3. Draw Bounding Boxes and Export Box as .csv
Use the following command to draw bounding boxes around detected traffic lights and export the box:

```bash
python draw_box_export.py sample\sample.mp4
```

### 4. Detect Traffic Lights in a Video with Bounding Boxes
Use the following command to detect traffic lights in a video and display bounding boxes:

```bash
python detect_video_in_box.py sample\sample.mp4
```

## Requirements

Make sure you have the following dependencies installed before running the scripts:

- Python 3.x
- OpenCV
- Pytorch
- NumPy

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

Special thanks to the contributors and the open-source community for their valuable resources and support.

For any issues or contributions, please open an issue or submit a pull request.

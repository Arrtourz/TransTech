### Project Execution Sequence and Operation Guide

-Download files from link under 'data' and 'models' folder.

#### 0. Adjust Detection Area Line
   - Run the `adjust_detection_area_line.py` script.
   - This script allows users to adjust the detection area line in the video.
   - Command:
     ```bash
     python /e:/Translab/carinfo/adjust_detection_area_line.py
     ```

#### 1. Vehicle Detection and Tracking
   - Run either the `vehicle_detection.py` or `vehicle_detection_4k.py` script.
   - These scripts detect and track vehicles in the video, saving screenshots of detected vehicles.
   - Command:
     ```bash
     python /e:/Translab/carinfo/vehicle_detection.py
     ```
     or
     ```bash
     python /e:/Translab/carinfo/vehicle_detection_4k.py
     ```

#### 2. License Plate Detection
   - Run the `license_plate_detection.py` script.
   - This script detects license plates within the saved vehicle screenshots and saves images of the license plate areas.
   - Command:
     ```bash
     python /e:/Translab/carinfo/license_plate_detection.py
     ```

#### 3. License Plate Recognition
   - Run either `license_plate_ocr.py` or `license_plate_ocr2.py`.
   - These scripts recognize the license plate numbers in the images of the detected license plate areas.
   - Command:
     ```bash
     python /e:/Translab/carinfo/license_plate_ocr.py
     ```
     or
     ```bash
     python /e:/Translab/carinfo/license_plate_ocr2.py
     ```

#### 4. Vehicle Information Recognition
   - Run the `gpt4v.py` script.
   - This script utilizes OpenAI’s GPT-4 model to recognize detailed vehicle information.
   - Command:
     ```bash
     python /e:/Translab/carinfo/gpt4v.py
     ```



Ensure that all dependent libraries are installed, e.g. cv2, numpy, easyocr, ultralytics, openai etc.
Adjust the file paths and parameters in the script as needed.
Before running the script, make sure the relevant model files (e.g. yolov8s_0.pt, best.pt) have been downloaded and placed in the correct paths.

import cv2
import numpy as np
from ultralytics import YOLO
import json
import argparse

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def point_in_quad(point, quad):
    x, y = point
    n = len(quad)
    inside = False
    p1x, p1y = quad[0]
    for i in range(n + 1):
        p2x, p2y = quad[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def main(video_path, box_file):
    # Load the YOLOv8 model
    model = YOLO('models/visDrone.pt')

    # Load quadrilateral data from JSON file
    box_data = load_json(box_file)
    quad1 = np.array(box_data['quad1'], dtype=np.int32)
    quad2 = np.array(box_data['quad2'], dtype=np.int32)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame, conf=0.3)  # Lowered confidence threshold

        # Reset counters for each frame
        vehicle_count_quad1 = 0
        vehicle_count_quad2 = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate center of the bottom edge of the bounding box
                center_x = (x1 + x2) // 2
                bottom_y = y2

                # Check if the vehicle is in either quadrilateral
                in_quad1 = point_in_quad((center_x, bottom_y), quad1)
                in_quad2 = point_in_quad((center_x, bottom_y), quad2)

                if in_quad1:
                    vehicle_count_quad1 += 1
                    color = (0, 255, 0)  # Green for quad1
                elif in_quad2:
                    vehicle_count_quad2 += 1
                    color = (255, 0, 0)  # Blue for quad2
                else:
                    color = (128, 128, 128)  # Gray for outside quads

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw quadrilaterals
        cv2.polylines(frame, [quad1], True, (0, 255, 0), 2)
        cv2.polylines(frame, [quad2], True, (255, 0, 0), 2)

        # Add current vehicle counts to the frame
        cv2.putText(frame, f'Quad1 Count: {vehicle_count_quad1}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Quad2 Count: {vehicle_count_quad2}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle Detection Script')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--box_file', type=str, default='area_box.json', help='Path to the box JSON file')
    
    args = parser.parse_args()

    main(args.video_path, args.box_file)
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('models/best6.pt')

def process_frame(frame, conf_threshold=0.3):
    # Perform detection
    results = model(frame, conf=conf_threshold)

    # Define color mapping
    color_map = {
        'red': (0, 0, 255),     # Red
        'yellow': (0, 255, 255),# Yellow
        'green': (0, 255, 0)    # Green
    }

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            conf = float(box.conf)

            class_name = model.names[cls]
            color = color_map.get(class_name, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            label_bg_left = x1
            label_bg_top = y1 - text_height - 5
            label_bg_right = label_bg_left + text_width + 5
            label_bg_bottom = label_bg_top + text_height + 5

            label_bg_top = max(label_bg_top, 0)
            label_bg_left = max(label_bg_left, 0)

            overlay = frame.copy()
            cv2.rectangle(overlay, (label_bg_left, label_bg_top), (label_bg_right, label_bg_bottom), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, label, (label_bg_left + 2, label_bg_bottom - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def main():
    video_path = 'sample/sample.mp4'
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_traffic_lights.mp4', fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        out.write(processed_frame)

        # Display the processed frame
        cv2.imshow('Traffic Light Detection', processed_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed. Output saved as 'output_traffic_lights.mp4'")

if __name__ == "__main__":
    main()

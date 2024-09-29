import cv2
import numpy as np
import json
import argparse

def adjust_lines(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return None, None
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Initialize lines
    line1 = np.array([(width//4, height//2), (3*width//4, height//2)], dtype=np.int32)
    line2 = np.array([(width//4, 2*height//3), (3*width//4, 2*height//3)], dtype=np.int32)
    
    selected_point = None
    selected_line = None

    def draw_lines():
        overlay = frame.copy()
        cv2.line(overlay, tuple(line1[0]), tuple(line1[1]), (0, 255, 0), 2)
        cv2.line(overlay, tuple(line2[0]), tuple(line2[1]), (0, 0, 255), 2)
        for i, line in enumerate([line1, line2]):
            for j, point in enumerate(line):
                cv2.circle(overlay, tuple(point), 5, (255, 0, 0), -1)
                cv2.putText(overlay, f'{i+1}-{j+1}', tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point, selected_line
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, line in enumerate([line1, line2]):
                for j, point in enumerate(line):
                    if np.linalg.norm(point - np.array([x, y])) < 10:
                        selected_point = j
                        selected_line = line
                        return
        elif event == cv2.EVENT_MOUSEMOVE:
            if selected_point is not None and selected_line is not None:
                selected_line[selected_point] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            selected_point = None
            selected_line = None

    cv2.namedWindow('Adjust Lines')
    cv2.setMouseCallback('Adjust Lines', mouse_callback)

    while True:
        display = draw_lines()
        cv2.imshow('Adjust Lines', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    return line1, line2

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Adjust detection lines for a video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', type=str, default='area_lines.json', help='Path to the output JSON file (default: detection_lines.json)')
    
    # Parse arguments
    args = parser.parse_args()

    # Call adjust_lines with the provided video path
    line1, line2 = adjust_lines(args.video_path)

    if line1 is None or line2 is None:
        print("Failed to adjust lines")
        return

    # Convert numpy arrays to lists for JSON serialization
    detection_lines = {
        "line1": line1.tolist(),
        "line2": line2.tolist()
    }

    # Save to JSON file
    with open(args.output, 'w') as f:
        json.dump(detection_lines, f, indent=4)

    print(f"Detection lines data saved to {args.output}")
    print("Line 1:")
    print(line1.tolist())
    print("Line 2:")
    print(line2.tolist())

if __name__ == "__main__":
    main()
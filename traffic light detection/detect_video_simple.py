import cv2
import numpy as np
import json
import os
import argparse

def adjust_box(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return None
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Initialize box (as a rectangle)
    box = np.array([(width//4, height//4), (3*width//4, height//4), 
                    (3*width//4, 3*height//4), (width//4, 3*height//4)], dtype=np.int32)
    
    selected_point = None

    def draw_box():
        overlay = frame.copy()
        cv2.polylines(overlay, [box], True, (0, 255, 0), 2)
        for i, point in enumerate(box):
            cv2.circle(overlay, tuple(point), 5, (255, 255, 0), -1)
            cv2.putText(overlay, f'{i+1}', tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(box):
                if np.linalg.norm(point - np.array([x, y])) < 10:
                    selected_point = i
                    return
        elif event == cv2.EVENT_MOUSEMOVE:
            if selected_point is not None:
                box[selected_point] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            selected_point = None

    cv2.namedWindow('Adjust Box')
    cv2.setMouseCallback('Adjust Box', mouse_callback)

    while True:
        display = draw_box()
        cv2.imshow('Adjust Box', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    return box

def export_to_json(box, output_path):
    box_dict = {
        "top_left": box[0].tolist(),
        "top_right": box[1].tolist(),
        "bottom_right": box[2].tolist(),
        "bottom_left": box[3].tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(box_dict, f, indent=4)
    
    print(f"Box coordinates exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Draw a box on a video frame and export coordinates to JSON.')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', type=str, help='Path to output JSON file (optional)')
    
    args = parser.parse_args()

    video_path = args.video_path
    result = adjust_box(video_path)

    if result is not None:
        print("Box coordinates:")
        print(result.tolist())
        
        if args.output:
            json_path = args.output
        else:
            # Generate default output JSON file path
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            json_path = os.path.join(video_dir, f"{video_name}_box.json")
        
        # Export to JSON
        export_to_json(result, json_path)
    else:
        print("Failed to adjust the box.")

if __name__ == "__main__":
    main()
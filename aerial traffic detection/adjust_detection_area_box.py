import cv2
import numpy as np
import json
import argparse

def adjust_quadrilaterals(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return None, None
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Initialize quadrilaterals
    quad1 = np.array([(width//4, height//2), (width//2, height//2), (width//3, height-100), (width//6, height-100)], dtype=np.int32)
    quad2 = np.array([(width//2, height//2), (3*width//4, height//2), (5*width//6, height-100), (2*width//3, height-100)], dtype=np.int32)
    
    selected_point = None
    selected_quad = None

    def draw_quads():
        overlay = frame.copy()
        cv2.polylines(overlay, [quad1], True, (0, 255, 0), 2)
        cv2.polylines(overlay, [quad2], True, (0, 0, 255), 2)
        for i, quad in enumerate([quad1, quad2]):
            for j, point in enumerate(quad):
                cv2.circle(overlay, tuple(point), 5, (255, 0, 0), -1)
                cv2.putText(overlay, f'{i+1}-{j+1}', tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point, selected_quad
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, quad in enumerate([quad1, quad2]):
                for j, point in enumerate(quad):
                    if np.linalg.norm(point - np.array([x, y])) < 10:
                        selected_point = j
                        selected_quad = quad
                        return
        elif event == cv2.EVENT_MOUSEMOVE:
            if selected_point is not None and selected_quad is not None:
                selected_quad[selected_point] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            selected_point = None
            selected_quad = None

    cv2.namedWindow('Adjust Quadrilaterals')
    cv2.setMouseCallback('Adjust Quadrilaterals', mouse_callback)

    while True:
        display = draw_quads()
        cv2.imshow('Adjust Quadrilaterals', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    return quad1, quad2

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Adjust detection area box for a video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', type=str, default='area_box.json', help='Path to the output JSON file (default: area_box.json)')
    
    # Parse arguments
    args = parser.parse_args()

    # Call adjust_quadrilaterals with the provided video path
    quad1, quad2 = adjust_quadrilaterals(args.video_path)

    if quad1 is None or quad2 is None:
        print("Failed to adjust quadrilaterals")
        return

    # Convert numpy arrays to lists for JSON serialization
    area_box = {
        "quad1": quad1.tolist(),
        "quad2": quad2.tolist()
    }

    # Save to JSON file
    with open(args.output, 'w') as f:
        json.dump(area_box, f, indent=4)

    print(f"Area box data saved to {args.output}")
    print("Quadrilateral 1:")
    print(quad1.tolist())
    print("Quadrilateral 2:")
    print(quad2.tolist())

if __name__ == "__main__":
    main()
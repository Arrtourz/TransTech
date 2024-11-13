import cv2
import numpy as np

def adjust_lines(video_path, max_display_width=1280, max_display_height=720):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create trackbar callback function
    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return
    
    # Get original frame dimensions
    orig_height, orig_width = frame.shape[:2]
    
    # Calculate scaling factor to fit within max display dimensions
    scale_width = min(1.0, max_display_width / orig_width)
    scale_height = min(1.0, max_display_height / orig_height)
    scale = min(scale_width, scale_height)
    
    # Calculate new dimensions
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Initialize lines with scaled coordinates
    line1 = np.array([(new_width//4, new_height//2), 
                      (3*new_width//4, new_height//2)], dtype=np.int32)
    line2 = np.array([(new_width//4, 2*new_height//3), 
                      (3*new_width//4, 2*new_height//3)], dtype=np.int32)
    
    selected_point = None
    selected_line = None

    def draw_lines(img):
        overlay = img.copy()
        cv2.line(overlay, tuple(line1[0]), tuple(line1[1]), (0, 255, 0), 2)
        cv2.line(overlay, tuple(line2[0]), tuple(line2[1]), (0, 0, 255), 2)
        for i, line in enumerate([line1, line2]):
            for j, point in enumerate(line):
                cv2.circle(overlay, tuple(point), 5, (255, 0, 0), -1)
                cv2.putText(overlay, f'{i+1}-{j+1}', tuple(point), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

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

    # Create windows
    cv2.namedWindow('Adjust Lines', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adjust Lines', new_width, new_height)
    cv2.setMouseCallback('Adjust Lines', mouse_callback)
    
    # Create trackbar
    cv2.createTrackbar('Frame', 'Adjust Lines', 0, total_frames-1, on_trackbar)
    
    # Add play/pause functionality
    playing = False
    current_frame = 0
    
    while True:
        if playing:
            current_frame = cv2.getTrackbarPos('Frame', 'Adjust Lines')
            current_frame = (current_frame + 1) % total_frames
            cv2.setTrackbarPos('Frame', 'Adjust Lines', current_frame)
        
        # Get current frame position
        current_frame = cv2.getTrackbarPos('Frame', 'Adjust Lines')
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Draw frame information
        time_sec = current_frame / fps
        info_text = f'Frame: {current_frame}/{total_frames-1} Time: {time_sec:.2f}s'
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        display = draw_lines(frame)
        cv2.imshow('Adjust Lines', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar to play/pause
            playing = not playing
        elif key == ord(','):  # Previous frame
            current_frame = max(current_frame - 1, 0)
            cv2.setTrackbarPos('Frame', 'Adjust Lines', current_frame)
        elif key == ord('.'):  # Next frame
            current_frame = min(current_frame + 1, total_frames-1)
            cv2.setTrackbarPos('Frame', 'Adjust Lines', current_frame)

    cv2.destroyAllWindows()
    cap.release()

    # Convert coordinates back to original scale
    scale_back = 1.0 / scale
    orig_line1 = (line1 * scale_back).astype(np.int32)
    orig_line2 = (line2 * scale_back).astype(np.int32)

    return orig_line1, orig_line2

# Usage
video_path = 'data/1_1_4k.mp4'
line1, line2 = adjust_lines(video_path)

print("Line 1:")
print(line1.tolist())
print("Line 2:")
print(line2.tolist())
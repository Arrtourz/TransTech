import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
from pathlib import Path

# Create a folder to save snapshots
snapshot_dir = Path('vehicle_snapshots')
snapshot_dir.mkdir(exist_ok=True)

# YOLOv8
model = YOLO('models/yolov8s_0.pt')
model = model.cuda()

# Define detection line for 4K resolution
line1 = np.array([[2196, 1248], [2736, 1221]], dtype=np.int32)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def save_vehicle_snapshot(frame, bbox, object_id, frame_count):
    """
    Save vehicle snapshot
    frame: original frame
    bbox: bounding box coordinates [x1, y1, x2, y2]
    object_id: vehicle ID
    frame_count: current frame count
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates do not exceed image boundaries
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Crop the image
    snapshot = frame[y1:y2, x1:x2]
    
    # Save the snapshot
    filename = f'vehicle_snapshots/vehicle_{object_id}_frame_{frame_count}.jpg'
    cv2.imwrite(filename, snapshot)
    print(f"Saved snapshot: {filename}")

    
class VehicleTracker:
    def __init__(self, max_disappear=5, max_distance=50, min_area=1000):
        self.nextID = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappear = max_disappear
        self.max_distance = max_distance
        self.min_area = min_area
        self.bboxes = {}
        self.areas = {}       # Store target area
        self.velocities = {}  # Store target velocity
        self.classes = {}
        self.scores = {}

    def get_area(self, bbox):
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def get_velocity(self, positions):
        """Calculate target velocity (pixels/frame)"""
        if len(positions) < 2:
            return None
        return np.array(positions[-1]) - np.array(positions[-2])

    def register(self, centroid, bbox, cls=None, score=None):
        self.objects[self.nextID] = deque([centroid], maxlen=20)
        self.bboxes[self.nextID] = bbox
        self.areas[self.nextID] = self.get_area(bbox)
        self.velocities[self.nextID] = None
        self.classes[self.nextID] = cls
        self.scores[self.nextID] = score
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bboxes[objectID]
        del self.areas[objectID]
        del self.velocities[objectID]
        del self.classes[objectID]
        del self.scores[objectID]

    def update(self, centroids, bboxes, classes=None, scores=None):
        if classes is None:
            classes = [None] * len(centroids)
        if scores is None:
            scores = [None] * len(centroids)

        # Filter out small area targets
        valid_indices = []
        for i, bbox in enumerate(bboxes):
            if self.get_area(bbox) >= self.min_area:
                valid_indices.append(i)
        
        centroids = [centroids[i] for i in valid_indices]
        bboxes = [bboxes[i] for i in valid_indices]
        classes = [classes[i] for i in valid_indices]
        scores = [scores[i] for i in valid_indices]

        if len(centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappear:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i], bboxes[i], classes[i], scores[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [self.objects[objectID][-1] for objectID in objectIDs]

            # Calculate distance matrix
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - np.array(centroids), axis=2)
            
            # Consider target velocity for prediction
            for i, objectID in enumerate(objectIDs):
                if self.velocities[objectID] is not None:
                    predicted_pos = objectCentroids[i] + self.velocities[objectID]
                    D[i] = np.linalg.norm(predicted_pos - np.array(centroids), axis=1)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue

                objectID = objectIDs[row]
                
                # Update target information
                self.objects[objectID].append(centroids[col])
                self.bboxes[objectID] = bboxes[col]
                self.areas[objectID] = self.get_area(bboxes[col])
                self.velocities[objectID] = self.get_velocity(self.objects[objectID])
                self.classes[objectID] = classes[col]
                self.scores[objectID] = scores[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            # Handle disappeared targets
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappear:
                        self.deregister(objectID)
            else:
                # Add new targets
                for col in unusedCols:
                    self.register(centroids[col], bboxes[col], classes[col], scores[col])

        return self.objects

def main():
    video_path = 'data//1_1_4k.mp4'
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")

    # Use 1080p display
    display_width = 1920
    display_height = 1080
    scale = min(display_width / frame_width, display_height / frame_height)
    display_width = int(frame_width * scale)
    display_height = int(frame_height * scale)

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', display_width, display_height)

    # Initialize tracker with optimized parameters
    tracker = VehicleTracker(max_disappear=5, max_distance=50, min_area=1000)
    vehicle_count_line1 = 0
    crossed_vehicles = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # YOLO detection
        results = model(frame, conf=0.7)  # Increase confidence threshold

        centroids = []
        bboxes = []
        classes = []
        scores = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate bottom center point as tracking point
                center_x = (x1 + x2) // 2
                bottom_y = y2
                
                centroids.append((center_x, bottom_y))
                bboxes.append([x1, y1, x2, y2])
                classes.append(int(box.cls[0]) if box.cls is not None else None)
                scores.append(float(box.conf[0]) if box.conf is not None else None)

        objects = tracker.update(centroids, bboxes, classes, scores)

        # Draw and process
        for (objectID, positions) in objects.items():
            center = positions[-1]
            prev_center = positions[0] if len(positions) > 1 else center

            if objectID not in crossed_vehicles:
                if intersect(line1[0], line1[1], prev_center, center):
                    vehicle_count_line1 += 1
                    crossed_vehicles.add(objectID)
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    
                    # Save high-quality snapshot
                    bbox = tracker.bboxes[objectID]
                    save_vehicle_snapshot(frame, bbox, objectID, frame_count)

            # Draw bounding box and information
            bbox = tracker.bboxes[objectID]
            color = (128, 128, 128) if objectID in crossed_vehicles else (0, 255, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Display detailed information
            info = f"ID:{objectID}"
            if tracker.scores[objectID] is not None:
                info += f" {tracker.scores[objectID]:.2f}"
            if tracker.velocities[objectID] is not None:
                vel = np.linalg.norm(tracker.velocities[objectID])
                info += f" v:{vel:.1f}"
            
            cv2.putText(frame, info, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw trajectory
            for i in range(1, len(positions)):
                if positions[i - 1] is None or positions[i] is None:
                    continue
                thickness = int(np.sqrt(len(positions) / float(i + 1)) * 2.5)
                cv2.line(frame, positions[i - 1], positions[i], (0, 255, 255), thickness)

        # Draw detection line
        cv2.line(frame, tuple(line1[0]), tuple(line1[1]), (0, 255, 0), 3)

        # Display count and progress
        cv2.putText(frame, f'Count: {vehicle_count_line1}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        progress = f'Frame: {frame_count}/{total_frames}'
        cv2.putText(frame, progress, (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Display and save
        display_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow('Frame', display_frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total vehicles counted: {vehicle_count_line1}")

if __name__ == "__main__":
    main()
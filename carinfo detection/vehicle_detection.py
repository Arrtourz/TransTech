import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os

# Create a folder to save snapshots
os.makedirs('vehicle_snapshots', exist_ok=True)

# YOLOv8
model = YOLO('models/yolov8s_0.pt')

# Move model to GPU device
model = model.cuda()

# Define detection line
line1 = np.array([[960, 649], [1413, 630]], dtype=np.int32)

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
    
    # Crop image
    snapshot = frame[y1:y2, x1:x2]
    
    # Save snapshot
    filename = f'vehicle_snapshots/vehicle_{object_id}_frame_{frame_count}.jpg'
    cv2.imwrite(filename, snapshot)
    print(f"Saved snapshot: {filename}")

class VehicleTracker:
    def __init__(self, max_disappear=10, max_distance=50):
        self.nextID = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappear = max_disappear
        self.max_distance = max_distance
        self.bboxes = {}  # Store bounding boxes

    def register(self, centroid, bbox):
        self.objects[self.nextID] = deque([centroid], maxlen=20)
        self.bboxes[self.nextID] = bbox  # Store bounding box
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bboxes[objectID]  # Delete bounding box information

    def update(self, centroids, bboxes):  # Add bboxes parameter
        if len(centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappear:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(centroids)):
                self.register(centroids[i], bboxes[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [self.objects[objectID][-1] for objectID in objectIDs]
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - np.array(centroids), axis=2)
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
                self.objects[objectID].append(centroids[col])
                self.bboxes[objectID] = bboxes[col]  # Update bounding box
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappear:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(centroids[col], bboxes[col])

        return self.objects

def main():
    video_path = 'data//1_1.mp4'
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate resized dimensions (set display window to half the original size)
    display_width = frame_width // 2
    display_height = frame_height // 2

    # Create named window and set it to be resizable
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', display_width, display_height)

    tracker = VehicleTracker(max_disappear=10, max_distance=50)
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
        results = model(frame, conf=0.7)

        centroids = []
        bboxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                bottom_y = y2
                centroids.append((center_x, bottom_y))
                bboxes.append([x1, y1, x2, y2])

        objects = tracker.update(centroids, bboxes)

        for (objectID, positions) in objects.items():
            center = positions[-1]
            prev_center = positions[0] if len(positions) > 1 else center

            if objectID not in crossed_vehicles:
                if intersect(line1[0], line1[1], prev_center, center):
                    vehicle_count_line1 += 1
                    crossed_vehicles.add(objectID)
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    print(f"Frame {frame_count}: Vehicle {objectID} crossed line 1")
                    
                    # Save vehicle snapshot using actual bounding box
                    save_vehicle_snapshot(frame, tracker.bboxes[objectID], objectID, frame_count)

            # Draw bounding box using actual bbox
            bbox = tracker.bboxes[objectID]
            color = (128, 128, 128) if objectID in crossed_vehicles else (0, 255, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, str(objectID), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trajectory
            for i in range(1, len(positions)):
                if positions[i - 1] is None or positions[i] is None:
                    continue
                cv2.line(frame, positions[i - 1], positions[i], (255, 255, 0), 2)

        # Draw detection line
        cv2.line(frame, tuple(line1[0]), tuple(line1[1]), (0, 255, 0), 2)

        # Display vehicle count
        cv2.putText(frame, f'Line1 Count: {vehicle_count_line1}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show and save frame
        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total vehicles crossed line 1: {vehicle_count_line1}")

if __name__ == "__main__":
    main()

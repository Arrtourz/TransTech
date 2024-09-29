import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import json
import argparse

# YOLOv8 model
model = YOLO('models/visDrone.pt')

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

class VehicleTracker:
    def __init__(self, max_disappear=10, max_distance=50):
        self.nextID = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappear = max_disappear
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextID] = deque([centroid], maxlen=5)
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, centroids):
        if len(centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappear:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(centroids)):
                self.register(centroids[i])
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
                    self.register(centroids[col])

        return self.objects

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main(video_path, lines_file):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    tracker = VehicleTracker(max_disappear=10, max_distance=50)
    vehicle_count_line1 = 0
    vehicle_count_line2 = 0
    crossed_vehicles_line1 = set()
    crossed_vehicles_line2 = set()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    lines_data = load_json(lines_file)
    line1 = np.array(lines_data['line1'], dtype=np.int32)
    line2 = np.array(lines_data['line2'], dtype=np.int32)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        results = model(frame, conf=0.3)

        centroids = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                bottom_y = (y1 + y2) // 2
                centroids.append((center_x, bottom_y))

        objects = tracker.update(centroids)

        for (objectID, positions) in objects.items():
            center = positions[-1]
            prev_center = positions[0] if len(positions) > 1 else center

            if objectID not in crossed_vehicles_line1:
                if intersect(line1[0], line1[1], prev_center, center):
                    vehicle_count_line1 += 1
                    crossed_vehicles_line1.add(objectID)
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    print(f"Frame {frame_count}: Vehicle {objectID} crossed line 1")

            if objectID not in crossed_vehicles_line2:
                if intersect(line2[0], line2[1], prev_center, center):
                    vehicle_count_line2 += 1
                    crossed_vehicles_line2.add(objectID)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    print(f"Frame {frame_count}: Vehicle {objectID} crossed line 2")

            x1, y1, x2, y2 = map(int, [center[0]-30, center[1]-30, center[0]+30, center[1]+30])
            color = (128, 128, 128) if objectID in crossed_vehicles_line1 or objectID in crossed_vehicles_line2 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, str(objectID), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(frame, tuple(line1[0]), tuple(line1[1]), (0, 255, 0), 2)
        cv2.line(frame, tuple(line2[0]), tuple(line2[1]), (0, 0, 255), 2)

        cv2.putText(frame, f'Line 1 Count: {vehicle_count_line1}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Line 2 Count: {vehicle_count_line2}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total vehicles detected on Line 1: {vehicle_count_line1}")
    print(f"Total vehicles detected on Line 2: {vehicle_count_line2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle Detection Script')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('lines_file', type=str, help='Path to the lines JSON file')
    
    args = parser.parse_args()

    main(args.video_path, args.lines_file)
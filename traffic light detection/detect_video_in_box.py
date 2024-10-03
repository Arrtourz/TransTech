import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import json
import os
import csv
from collections import defaultdict
import io
# 加载YOLOv8模型
model = YOLO('models/best6.pt')

# 初始化区域状态
area_state = "Unknown"

# 用于记录状态变化的变量
state_timeline = []
current_state_start = 0
frame_count = 0

def load_detection_box(video_path):
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(video_dir, f"{video_name}_box.json")
    
    try:
        with open(json_path, 'r') as f:
            box_data = json.load(f)
        
        detection_box = np.array([
            box_data["top_left"],
            box_data["top_right"],
            box_data["bottom_right"],
            box_data["bottom_left"]
        ], dtype=np.int32)
        
        print(f"Successfully loaded detection box from {json_path}")
        return detection_box
    except FileNotFoundError:
        print(f"Warning: {json_path} not found. Using default detection box.")
        return np.array([[516, 469], [658, 469], [657, 572], [510, 563]], dtype=np.int32)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_path}. Using default detection box.")
        return np.array([[516, 469], [658, 469], [657, 572], [510, 563]], dtype=np.int32)

def point_in_quad(point, quad):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(point, quad[0], quad[1])
    d2 = sign(point, quad[1], quad[2])
    d3 = sign(point, quad[2], quad[3])
    d4 = sign(point, quad[3], quad[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

    return not (has_neg and has_pos)

def process_frame(frame, detection_box, conf_threshold=0.3):
    global area_state, current_state_start, frame_count, state_timeline
    
    frame_count += 1
    # 执行检测
    results = model(frame, conf=conf_threshold)

    # 定义颜色映射
    color_map = {
        'red': (0, 0, 255),     # 红色
        'yellow': (0, 255, 255),# 黄色
        'green': (0, 255, 0)    # 绿色
    }

    new_state = "Unknown"

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if point_in_quad(center_point, detection_box):
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
                
                # 更新状态
                new_state = class_name

    # 检查状态是否发生变化
    if new_state != area_state:
        if area_state != "Unknown":
            state_timeline.append((area_state, current_state_start, frame_count - current_state_start))
        area_state = new_state
        current_state_start = frame_count

    # 绘制检测区域和状态
    cv2.polylines(frame, [detection_box], True, (255, 255, 255), 2)
    cv2.putText(frame, "Detection Area", tuple(detection_box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 在视频底部添加状态信息
    status_text = f"Traffic Light State: {area_state}"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(area_state, (255, 255, 255)), 2)

    return frame


def save_timeline(timeline, fps, output_file):
    csv_content = io.StringIO()
    csvwriter = csv.writer(csv_content)
    csvwriter.writerow(['State', 'Start Time', 'Duration', 'End Time'])
    for state, start_frame, duration in timeline:
        start_time = start_frame / fps
        duration_time = duration / fps
        end_time = start_time + duration_time
        csvwriter.writerow([state, f"{start_time:.2f}s", f"{duration_time:.2f}s", f"{end_time:.2f}s"])
    
    csv_str = csv_content.getvalue()
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            csvfile.write(csv_str)
        print(f"Successfully saved timeline to {output_file}")
    except Exception as e:
        print(f"Error saving timeline to CSV: {e}")
    
    print("\nCSV Content:")
    print(csv_str)
    return csv_str

def main(video_path):
    global current_state_start, frame_count, state_timeline

    detection_box = load_detection_box(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video properties: {frame_width}x{frame_height} at {fps} fps")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_traffic_light_detection.mp4', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, detection_box)

        cv2.imshow('Traffic Light Detection', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 添加最后一个状态到时间轴
    if area_state != "Unknown":
        state_timeline.append((area_state, current_state_start, frame_count - current_state_start))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")
    print(f"Final state timeline: {state_timeline}")

    # 保存时间轴到CSV文件并输出内容
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(os.path.dirname(video_path), f"{video_name}_traffic_light_timeline.csv")
    csv_content = save_timeline(state_timeline, fps, csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traffic Light Detection in a Single Box with Timeline')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    args = parser.parse_args()
    
    main(args.video_path)
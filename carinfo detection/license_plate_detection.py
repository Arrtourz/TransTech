import cv2
from ultralytics import YOLO
import os

class LicensePlateDetector:
    def __init__(self):
        self.np_model = YOLO('models//best.pt')
        self.conf_threshold = 0.5  # Significantly lower detection threshold

    def detect_license_plate(self, image_path):
        """Detect license plates in the image and save the license plate region"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Save original detection results
        detections = self.np_model(image, conf=self.conf_threshold)[0]
        
        # Print all detection results
        print("\nDetection results:")
        for i, box in enumerate(detections.boxes):
            conf = box.conf.item()
            coords = box.xyxy[0].tolist()
            print(f"Detection {i+1}:")
            print(f"Confidence: {conf:.3f}")
            print(f"Bounding box: {coords}")
        
        # If there are multiple detection results, only take the one with the highest confidence
        if len(detections.boxes) > 0:
            # Get the detection result with the highest confidence
            best_box = max(detections.boxes, key=lambda x: x.conf)
            plate_score = best_box.conf.item()
            plate_x1, plate_y1, plate_x2, plate_y2 = map(int, best_box.xyxy[0].tolist())
            
            # Extract the license plate region
            plate_roi = image[plate_y1:plate_y2, plate_x1:plate_x2]
            
            # Save the license plate ROI for debugging
            image_name = os.path.basename(image_path)
            plate_roi_path = f'output//plate_roi_{image_name}'
            cv2.imwrite(plate_roi_path, plate_roi)
            
            result = {
                'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                'score': plate_score,
                'plate_roi_path': plate_roi_path
            }
        else:
            print("No license plate detected")
            result = None
        
        return result

def main():
    try:
        detector = LicensePlateDetector()
        folder_path = "vehicle_snapshots1_1_4k"
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                result = detector.detect_license_plate(image_path)
                
                if result:
                    print(f"\nDetection results for {filename}:")
                    print(f"Detection confidence: {result['score']:.3f}")
                    print(f"License plate region image: {result['plate_roi_path']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

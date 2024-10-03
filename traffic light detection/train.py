# Load a pretrained YOLOv8n model from Ultralytics
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


# Train the model on our custom dataset
yaml_file_path = "datasets/data.yaml"

results = model.train(
    data=yaml_file_path,     # Path to the dataset configuration file. 
    epochs=100,              # Number of epochs to train for
    imgsz=640,               # Size of input images as integer
    device=0,                # Device to run on, i.e. cuda device=0 
    patience=50,             # Epochs to wait for no observable improvement for early stopping of training
    batch=32,                # Number of images per batch
    optimizer='auto',        # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0=0.0001,              # Initial learning rate 
    lrf=0.1,                 # Final learning rate (lr0 * lrf)
    dropout=0.1,             # Use dropout regularization
    seed=0                   # Random seed for reproducibility
)
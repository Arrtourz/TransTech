
# Load a pretrained YOLOv8n model from Ultralytics
from ultralytics import YOLO

model = YOLO('models/best6.pt')

# Run inference on the image
results = model('sample/sample.jpg',conf=0.3)


img = results[0].plot()
from PIL import Image
Image.fromarray(img).save("output.jpg")
print("Results saved to output.jpg")
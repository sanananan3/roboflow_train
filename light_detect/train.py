from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11s.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="/home/kist/roboflow/light_detect/data.yaml",  # Path to dataset configuration file
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size for training
    batch=32,
    device="cuda",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()


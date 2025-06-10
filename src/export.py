from ultralytics import YOLO

# Load a model
model = YOLO("model/welding_cls_best.pt")  # load an official model

# Export the model
model.export(format="onnx",dynamic=True)
from ultralytics import YOLO

# Load a model
model = YOLO("/home/dxc/code/ai_exam/weights/welding2_k2/welding_switch_last.pt")  # load an official model

# Export the model
model.export(format="onnx",dynamic=True)
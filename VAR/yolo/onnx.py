# import torch
# import torch.onnx
# from yolo.common import DetectMultiBackend  # Pastikan ini sesuai dengan struktur YOLO

# # Load the PyTorch model\
# model = DetectMultiBackend('C:\\Users\\LENOVO\\grup\\VAR\\best.pt')  # Gantilah dengan model YOLO Anda
# model.eval()

# # Create a dummy input tensor matching the input size expected by the model
# dummy_input = torch.randn(1, 3, 640, 640)  # Ubah sesuai dengan ukuran input model

# # Export the model to ONNX
# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=['input'], output_names=['output'])

# import sys
# sys.path.append('C:/Users/LENOVO/grup/VAR/yolo/yolov5')  # Gantilah path dengan path ke repositori YOLOv5 Anda

# from models.common import DetectMultiBackend

from pathlib import Path

# Unduh model YOLOv5s
weights_url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'
weights_path = Path('C:\\Users\\LENOVO\\grup\\VAR\\best.pt')
if not weights_path.exists():
    import requests
    r = requests.get(weights_url)
    weights_path.write_bytes(r.content)


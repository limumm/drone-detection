from ultralytics import YOLO
# 设置相对路径
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO('yolov8n.pt')
print(current_dir + '/dataset/data.yaml')
# model.train(task='detect', data=current_dir + '/dataset/data.yaml', epochs=100, batch=32, imgsz=640, resume=False, workers=25, device='0', save_period=5)
model.train(model='./runs/detect/train53/weights/last.pt',task='detect', data=current_dir + '/dataset/data.yaml', epochs=50, batch=32, imgsz=640, resume=False, workers=25, device='0', save_period=5)

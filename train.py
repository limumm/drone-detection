from ultralytics import YOLO
# 设置相对路径
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO('./weights/yolov8n.pt')
# print(current_dir + '/dataset/data.yaml')
# model.train(task='detect', data=current_dir + '/dataset/data.yaml', epochs=100, batch=32, imgsz=640, resume=False, workers=25, device='0', save_period=5)
model.train(task='detect', data=os.path.join(current_dir, 'data.yaml'), epochs=1, batch=32, imgsz=640, resume=False, workers=25, device='0', save_period=1)

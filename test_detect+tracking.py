import cv2
from ultralytics import YOLO

model = YOLO('./weights/best.pt')

import os
import json
root_dir = os.path.dirname(os.path.abspath(__file__))
test_folder = os.path.join(root_dir, '无人机检测与追踪/test')
test_subfolder = os.listdir(test_folder)
test_subfolder.sort()
os.makedirs(os.path.join(root_dir, 'test_res'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'test_res', 'detect+tracking_res'), exist_ok=True)

for test_sub in test_subfolder:
    frame_files = os.listdir(os.path.join(test_folder, test_sub))
    frame_files.sort()
    rect_list = []
    with open(os.path.join(test_folder, test_sub, 'IR_label.json'), 'r') as f:
        data = json.load(f)
    prior_bbox = data['res'][0]
    for _, frame_file in enumerate(frame_files):
        frame_path = os.path.join(test_folder, test_sub, frame_file)
        frame = cv2.imread(frame_path)
        # 新建一个test_res/detect+tracking_res/test_sub文件夹
        # os.makedirs(os.path.join(root_dir, 'test_res', 'detect+tracking_res', test_sub), exist_ok=True)
        if frame is not None:
            # 在其他帧中使用YOLOv8的预测结果
            results = model.track(frame, persist=True)
            xyxy = results[0].boxes.xyxy
            xywh = results[0].boxes.xywh
            # 画出bbox
            for i in range(len(xyxy)):
                x, y, _, _ = xyxy[i].int().tolist()
                _, _, w, h = xywh[i].int().tolist()
                bbox = [x, y, w, h]
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # 保存画出bbox的frame
            # cv2.imwrite(os.path.join('./test_res/detect+tracking_res', test_sub, frame_file), frame)
            if len(xyxy) != 0:
                rect_list.append(bbox)
            else:
                rect_list.append([])
        with open(os.path.join(test_folder, test_sub, 'IR_label.json'), 'r') as f:
            ir_label = json.load(f)
        if len(ir_label['res']) != 0:
            rect_list[0] = ir_label['res'][0]
        # 将exist_list和rect_list制作成json
        data = {"res":rect_list}
        json_string = json.dumps(data)
        with open(os.path.join(root_dir, 'test_res', 'detect+tracking_res', test_sub+'.txt'), 'w') as f:
            f.write(json_string)
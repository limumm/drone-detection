import cv2
from ultralytics import YOLO

model = YOLO('./runs/detect/train53/weights/best.pt')

import os
import json
test_folder = '/media/limumu/ADC4EA8E9E19D40E/DDProj/无人机检测与追踪/test/'
test_subfolder = os.listdir(test_folder)
test_subfolder.sort()
if not os.path.exists('./test_res'):
    os.mkdir('./test_res')
if not os.path.exists('./test_res/detect+tracking_res2'):
    os.mkdir('./test_res/detect+tracking_res2')

for test_sub in ['20190925_141417_1_3']:
    frame_files = os.listdir(test_folder+ "/" +test_sub)
    frame_files.sort()
    rect_list = []
    import json

    with open(os.path.join(test_folder, test_sub, 'IR_label.json'), 'r') as f:
        data = json.load(f)
    prior_bbox = data['res'][0]
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(test_folder, test_sub, frame_file)
        frame = cv2.imread(frame_path)
        # 新建一个test_res/detect+tracking_res/test_sub文件夹
        if not os.path.exists(os.path.join('./test_res/detect+tracking_res2', test_sub)):
            os.mkdir(os.path.join('./test_res/detect+tracking_res2', test_sub))
        # os.mkdir(os.path.join('./test_res/detect+tracking_res', test_sub))
        if frame is not None:
            if i == 0:
                # 在第一帧中使用手动标注的目标位置作为先验知识
                xywh = [prior_bbox]
                results = model.track(frame, persist=True)
            else:
                # 在其他帧中使用YOLOv8的预测结果
                results = model.track(frame, persist=True)
            xyxy = results[0].boxes.xyxy
            xywh = results[0].boxes.xywh
            # 画出bbox
            for i in range(len(xyxy)):
                x, y, _, _ = xyxy[i].int().tolist()
                _, _, w, h = xywh[i].int().tolist()
                bbox = [x, y, w, h]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # 保存画出bbox的frame
            cv2.imwrite(os.path.join('./test_res/detect+tracking_res2', test_sub, frame_file), frame)
            if len(xyxy) != 0:
                rect_list.append(bbox)
            else:
                rect_list.append([])
        # 将exist_list和rect_list制作成json
        data = {"res":rect_list}
        json_string = json.dumps(data)
        with open('./test_res/detect+tracking_res2/' + test_sub +  '/'+ test_sub + '.json', 'w') as f:
            f.write(json_string)
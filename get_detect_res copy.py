import cv2
from ultralytics import YOLO
import os
import json
model = YOLO('./runs/detect/train53/weights/best.pt')

test_folder = '/media/limumu/ADC4EA8E9E19D40E/DDProj/无人机检测与追踪/test/'
test_subfolder = os.listdir(test_folder)
test_subfolder.sort()
if not os.path.exists('./test_res'):
    os.mkdir('./test_res')
if not os.path.exists('./test_res/detect_res'):
    os.mkdir('./test_res/detect_res')
for test_sub in test_subfolder:
    test_sub_path = os.path.join(test_folder, test_sub)
    frame_files = os.listdir(test_sub_path)
    frame_files = [f for f in frame_files if f.endswith('.jpg')]
    frame_files.sort()
    rect_list = []
    # results = model.predict(source=test_sub[os.path.join(test_sub_path, frame_file) for frame_file in frame_files], stream=True)
    results = model.predict(source=test_sub_path, stream=True)
    for result in results:
        if not os.path.exists(os.path.join('./test_res/detect_res', test_sub)):
            os.mkdir(os.path.join('./test_res/detect_res', test_sub))
        file_name = result.path.split('/')[-1]
        boxes = result.boxes
        xyxy = boxes.xyxy
        xywh = boxes.xywh
        if len(xyxy) != 0:
            x, y, _, _ = boxes.xyxy[0].int().tolist()
            _, _, w, h = boxes.xywh[0].int().tolist()
            bbox = [x, y, w, h]
            rect_list.append(bbox)
            frame = result.orig_img
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.imwrite(os.path.join('./test_res/detect_res', test_sub, file_name), frame)
        else:
            cv2.imwrite(os.path.join('./test_res/detect_res', test_sub, file_name), result.orig_img)
            rect_list.append([])
        
        
        
    data = {"res":rect_list}
    json_string = json.dumps(data)
    with open('./test_res/detect_res/' + test_sub +  '/'+ test_sub + '.json', 'w') as f:
        f.write(json_string)
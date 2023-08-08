import os
import json
import shutil
from multiprocessing import Pool, Manager
# 创建dataset文件夹
if not os.path.exists('dataset/train'):
    os.mkdir('dataset/train')
if not os.path.exists('dataset/train/imgs'):
    os.mkdir('dataset/train/imgs')
if not os.path.exists('dataset/train/labels'):
    os.mkdir('dataset/train/labels')
def process_folder(source_folder, target_folder, image_count):
    
    json_file_path = os.path.join(source_folder, "IR_label.json")
    label_folder = os.path.join(target_folder, "labels")
    image_folder_path = os.path.join(target_folder, "imgs")
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
        exist_list = data["exist"]
        gt_rect_list = data["gt_rect"]

    for i, (exist, gt_rect) in enumerate(zip(exist_list, gt_rect_list), 1):
        image_name = str(image_count.value).zfill(6) + ".jpg"
        txt_name = str(image_count.value).zfill(6) + ".txt"
        print(f"Processing {image_name}")
        image_count.value += 1

        image_path = os.path.join(source_folder, f"{i:06}.jpg")
        target_image_path = os.path.join(image_folder_path, image_name)

        shutil.copy(image_path, target_image_path)

        if exist == 1:
            txt_file_path = os.path.join(label_folder, txt_name)
            with open(txt_file_path, "w") as txt_file:
                x_center = (gt_rect[0] + gt_rect[2] / 2) / 640
                y_center = (gt_rect[1] + gt_rect[3] / 2) / 512
                width = gt_rect[2] / 640
                height = gt_rect[3] / 512
                txt_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    
    source_folder = './无人机检测与追踪/train/'
    train_path = './dataset/train/'

    if not os.path.exists(train_path):
        os.makedirs(train_path)
        
    folders = [os.path.join(source_folder, folder_name) for folder_name in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, folder_name))]
    
    manager = Manager()
    image_count = manager.Value('i', 1)

    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(process_folder, [(folder, train_path, image_count) for folder in folders])
import os
import json
import shutil
import numpy as np
from PIL import Image

def resize_and_process_image(image_path, target_image_path, gt_rect, label_file_path):
    image = Image.open(image_path)
    width, height = image.size

    # Resize the image to 640x640
    new_width, new_height = 640, 640
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the new bounding box positions based on the resized image
    x_center = (gt_rect[0] + gt_rect[2] / 2) / width
    y_center = (gt_rect[1] + gt_rect[3] / 2) / height
    width_ratio = new_width / width
    height_ratio = new_height / height

    new_x_center = x_center * width_ratio
    new_y_center = y_center * height_ratio
    new_width = gt_rect[2] * width_ratio
    new_height = gt_rect[3] * height_ratio

    with open(label_file_path, "w") as txt_file:
        txt_file.write(f"0 {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")

    resized_image.save(target_image_path)
def process_images(source_folder, target_folder):
    image_count = 1
    label_folder = os.path.join(target_folder, "labels")
    image_folder_path = os.path.join(target_folder, "imgs")
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    for folder_name in sorted(os.listdir(source_folder)):
        folder_path = os.path.join(source_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        json_file_path = os.path.join(folder_path, "IR_label.json")
        

        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            exist_list = data["exist"]
            gt_rect_list = data["gt_rect"]

            for i, (exist, gt_rect) in enumerate(zip(exist_list, gt_rect_list), 1):
                image_name = str(image_count).zfill(6) + ".jpg"
                txt_name = str(image_count).zfill(6) + ".txt"
                image_count += 1
                print('img_count ' + str(image_count))
                image_path = os.path.join(folder_path, f"{i:06}.jpg")
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
# 随机划分训练集和验证集
def split_train_val(train_path, val_path, val_rate=0.2):
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    train_imgs_path = os.path.join(train_path, "imgs")
    train_labels_path = os.path.join(train_path, "labels")
    val_imgs_path = os.path.join(val_path, "imgs")
    val_labels_path = os.path.join(val_path, "labels")
    if not os.path.exists(val_imgs_path):
        os.makedirs(val_imgs_path)
    if not os.path.exists(val_labels_path):
        os.makedirs(val_labels_path)
    img_names = os.listdir(train_imgs_path)
    img_count = len(img_names)
    val_count = int(img_count * val_rate)
    val_indices = np.random.choice(img_count, val_count, replace=False)
    for i, img_name in enumerate(img_names):
        if i in val_indices:
            img_path = os.path.join(train_imgs_path, img_name)
            label_path = os.path.join(train_labels_path, img_name[:-4] + ".txt")
            shutil.move(img_path, val_imgs_path)
            if os.path.exists(label_path):
                shutil.move(label_path, val_labels_path)
        # else:
        #     shutil.copy(img_path, val_imgs_path)
        #     shutil.copy(label_path, val_labels_path)
        print(f"processing {i + 1} / {img_count}")

if __name__ == "__main__":
    
    source_folder = './无人机检测与追踪/train/'
    train_path = './dataset/train/'

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    # process_images(source_folder, train_path)

    val_path = './dataset/val/'
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    split_train_val(train_path, val_path, val_rate=0.2)
    
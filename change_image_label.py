from PIL import Image
import os

def fill_with_black(input_path, output_path, target_size):
    # Open the image
    image = Image.open(input_path)
    
    # Create a new blank image with the target size and fill it with black
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    
    # Calculate the position to paste the original image (left-top aligned)
    paste_position = (0, 0)
    
    # Paste the original image on the black canvas
    new_image.paste(image, paste_position)
    
    # Save the resulting image
    new_image.save(output_path)

def update_height_in_txt(input_path):
    # Read the content of the txt file
    with open(input_path, 'r') as file:
        content = file.read().strip()

    # Split the content by whitespace and get the last number
    numbers = content.split()
    # last_number = float(numbers[-1])
    y_new = float(numbers[2])
    # Multiply the last number by 512/640
    # updated_last_number = last_number * (512 / 640)
    updated_y_new = y_new * (512 / 640)
    # Update the last number in the numbers list
    # numbers[-1] = str(updated_last_number)
    numbers[2] = str(updated_y_new)
    # Join the numbers back into a single string
    updated_content = ' '.join(numbers)
    # Write the updated content back to the txt file
    with open(input_path, 'w') as file:
        file.write(updated_content)

# # Folder path containing the images
# input_folder = './dataset/images/train'
# # Folder path to save the processed images
# output_folder = './dataset/images/train'
# # Target size for the processed images
# target_size = (640, 640)

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Process each image in the input folder
# for filename in os.listdir(input_folder):
#     input_path = os.path.join(input_folder, filename)
#     output_path = os.path.join(output_folder, filename)
#     fill_with_black(input_path, output_path, target_size)

txt_folder = './dataset/labels/train'
for filename in os.listdir(txt_folder):
    file_path = os.path.join(txt_folder, filename)
    update_height_in_txt(file_path)
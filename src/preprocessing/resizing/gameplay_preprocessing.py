import os
from PIL import Image
from torchvision import transforms

source_root = 'data/raw/gameplay/'
destination_folder = 'data/processed/resized_combined/'

# List of specific subfolders you want to process
subfolders = ["Among Us", "Apex Legends", "Fortnite", "Forza Horizon", "Free Fire", "Genshin Impact"]

# Ensure destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Define transformations for InceptionV3
transform = transforms.Compose([
    transforms.Resize(299),  # Resize the image
    transforms.CenterCrop(299)  # Crop the center of the image
])

# 1. Traverse directories
for dirpath, dirnames, filenames in os.walk(source_root):
    # Only process directories in the subfolders list
    if os.path.basename(dirpath) in subfolders:
        prefix = os.path.basename(dirpath)[:4]  # Extract the first 4 letters
        for file_name in filenames:
            # 2. Identify PNG images
            if file_name.endswith('.png'):
                image_path = os.path.join(dirpath, file_name)
                
                # Open the image
                img = Image.open(image_path)
                
                # 3. Convert the image
                img_transformed = transform(img)
                
                # Convert the image to RGB mode
                img_transformed = img_transformed.convert('RGB')

                # Name the converted image (prepend the prefix and change extension to ".jpg")
                base_name = os.path.splitext(file_name)[0]
                converted_image_name = prefix + "_" + base_name + ".jpg"
                
                # 4. Move to destination folder
                img_transformed.save(os.path.join(destination_folder, converted_image_name))
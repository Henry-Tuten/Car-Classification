import os
import random
import shutil
from PIL import Image
from torchvision import transforms


# Open folder with raw cars
# 60,000+ car images jpg
# Mark all car images as true
# 1. Navigate to the folder
source_folder = 'data/raw/cars/'
destination_folder = 'data/processed/resized_combined/'

# Check if destination folder exists, otherwise create it
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 2. Randomly select 10,000 jpg images
all_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
selected_files = random.sample(all_files, 5650)

# 3. Convert these images to the format expected by InceptionV3
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299)
])

for file_name in selected_files:
    image_path = os.path.join(source_folder, file_name)
    img = Image.open(image_path)
    
    # Apply transformations
    img_transformed = transform(img)
    
    # Save the transformed image to the destination folder
    img_transformed.save(os.path.join(destination_folder, file_name))

# Open folder with gameplay images and take all of them
# 10,011 gameplay images 640 x 360 png
# Mark all art images as false



# Make sure there are an even number of samples

# Resize all images to the same dimensions

# Save to processed data

# Join all three together for training 



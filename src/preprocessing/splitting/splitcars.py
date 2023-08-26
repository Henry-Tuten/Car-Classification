import os
import shutil
import random

# Define the directories
source_directory = 'data/processed/resized_cars/'
target_directory1 = 'data/processed/train_images/cars/'
target_directory2 = 'data/processed/val_images/cars/'

# Ensure target directories exist
if not os.path.exists(target_directory1):
    os.makedirs(target_directory1)

if not os.path.exists(target_directory2):
    os.makedirs(target_directory2)

# List all files in the source directory
all_files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

# Shuffle the list to get random files
random.shuffle(all_files)

# Calculate the split index
split_at = int(len(all_files) * 0.9)

# Split the files into two lists
target_files1 = all_files[:split_at]
target_files2 = all_files[split_at:]

# Move the files to their respective target directories
for file in target_files1:
    shutil.move(os.path.join(source_directory, file), os.path.join(target_directory1, file))

for file in target_files2:
    shutil.move(os.path.join(source_directory, file), os.path.join(target_directory2, file))
import os
import csv

source_root = 'data/processed_combined/'
destination_file = 'data/processed/labeled_images5050.csv'

# List of specific subfolders you want to process
subfolders = ["Among Us", "Apex Legends", "Fortnite", "Forza Horizon", "Free Fire", "Genshin Impact"]

# Check if CSV file is empty or doesn't exist to write the header
need_header = not os.path.exists(destination_file) or os.path.getsize(destination_file) == 0

# Open CSV file for appending
with open(destination_file, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # If needed, write header row
    if need_header:
        csv_writer.writerow(["Image Name", "Label"])
    
    # 1. Traverse directories
    for dirpath, dirnames, filenames in os.walk(source_root):
        # Only process directories in the subfolders list
        if os.path.basename(dirpath) in subfolders:
            prefix = os.path.basename(dirpath)[:4]  # Extract the first 4 letters
            for file_name in filenames:
                # 2. Identify PNG images
                if file_name.endswith('.png'):
                    # Name the image (prepend the prefix)
                    image_name = prefix + "_" + file_name
                    
                    # 3. Write image name and label to CSV
                    csv_writer.writerow([image_name, "False"])
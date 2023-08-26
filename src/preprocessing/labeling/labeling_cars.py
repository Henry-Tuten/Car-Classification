import os
import csv

# Define paths
source_folder = 'data/processed/resized_combined/'
output_csv_path = 'data/processed/labeled_images5050.csv'

with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['filename', 'label'])  # header row

    for img_name in os.listdir(source_folder):
        csvwriter.writerow([img_name, 'True'])
from pathlib import Path
import csv

base_dir = Path('/media/local-admin/Crucial X92/MAIA/HES/data_3_4/HCC_test')

with open('/home/local-admin/Documents/projects/CLAM/dataset_csv/HCC_test.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['slide_name'])  # Write header

    for p in base_dir.iterdir():
        if p.is_dir():
            writer.writerow([p.name])  # Write slide name
import pandas as pd
import os
import wget

image_data = pd.read_csv('MYSQL_products_RAW_202402042150.csv')
os.makedirs('cs-images', exist_ok=True)
imgs = image_data.iloc[:,-1].tolist()
imgs = [imgs[i].split(';')[-1] for i in range(len(imgs))]

def download_file(url, local_dir):
    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # Determine the local file path
    local_file_path = os.path.join(local_dir, os.path.basename(url))
    
    # Download the file using wget
    wget.download(url, local_file_path)

for img in imgs:
    shop = img.split('/')[3]
    local_dir = f'cs-images/{shop}'
    download_file(img, local_dir)

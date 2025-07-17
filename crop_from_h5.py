import cv2
import time
from PIL import Image
import os
import pandas as pd
import h5py
import openslide
import argparse
import openslide
from tqdm import tqdm
from PIL import ImageFilter

from dataset import *

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


parser = argparse.ArgumentParser(description='Crop WSI')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--crop_path', type=str, default=None)
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--resolution_level', type=int, default=0)
parser.add_argument('--scale_prefixed', type=int, default=-1)
args = parser.parse_args()

if __name__ == '__main__':

    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError
    
    bags_dataset = Dataset_All_Bags(csv_path)
    total = len(bags_dataset)

    wsi_infos = []

    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

		# slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext) ##### Commented

		### replaced by :
        g = os.walk(args.data_slide_dir)

        for dirpath, dirnames, filenames in g:
            for filename in filenames:
                if filename == slide_id+args.slide_ext:
                    slide_crop_path = os.path.join(args.crop_path, slide_id)
                    create_dir(slide_crop_path)
                    slide_file_path = os.path.join(dirpath, filename)

                    slide = openslide.OpenSlide(slide_file_path)

                    with h5py.File(h5_file_path, 'r') as f:
                        n_patches = len(f['coords'])
                        for patch_idx in range(0, n_patches, 10):
                            coord = f['coords'][patch_idx]
                            # patch_size = f['coords'].attrs['patch_size']

                            img_0 = slide.read_region(coord, args.resolution_level, (args.target_patch_size*np.abs(args.scale_prefixed), args.target_patch_size*np.abs(args.scale_prefixed))).convert('RGB')      
                            img = Image.new("RGB", img_0.size)                        
                            img = img_0
                            img = img.resize((args.target_patch_size, args.target_patch_size), Image.BILINEAR)

                            img.save(slide_crop_path + '/' + str(patch_idx) + '.jpg', quality = 100)
                            # img.save(slide_crop_path + '/' + str(patch_idx) + '.png')

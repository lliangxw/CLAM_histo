import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
from pathlib import Path

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

import csv

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0, patch_size = None):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):  ####
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			# if patch_size != None:
			# 	features = model(batch, patch_size)
			# else:
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}

			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
# parser.add_argument('--train_set_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'histoCAE', 'histoCAE_latent'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=256)
parser.add_argument('--scale_prefixed', type=int, default=-1)
args = parser.parse_args()

test_set_ids = []  ### to be defined prior to running the script

with open('/home/local-admin/Documents/projects/CLAM/test_csv/CCA_vs_HCC_256_SimCLR_all_split.csv', newline="", encoding='utf-8') as f:
	reader = csv.DictReader(f)
	for row in reader:
		test_set_ids.append(row['test'])

if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}

	# base_dir = Path(args.train_set_dir)

	# train_set_names = [p.name for p in base_dir.iterdir() if p.is_dir()] ### Added, not all WSIs are used for training

	for bag_candidate_idx in tqdm(range(total)):
		slide_file_path = None
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		# if slide_id in train_set_names:
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

			# slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext) ##### Commented

			### replaced by :

		if slide_id in test_set_ids:   #### test_set_ids to be defined prior to running the script

			g = os.walk(args.data_slide_dir)

			for dirpath, dirnames, filenames in g:
				for filename in filenames:
					if filename == slide_id+args.slide_ext:
						slide_file_path = os.path.join(dirpath, filename)
						
			#####

			if slide_file_path != None:
				print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
				print(slide_id, slide_file_path)

				if not args.no_auto_skip and slide_id+'.pt' in dest_files:
					print('skipped {}'.format(slide_id))
					continue 

				output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
				time_start = time.time()
				wsi = openslide.open_slide(slide_file_path)
				dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
											wsi=wsi, 
											img_transforms=img_transforms, 
											scale_prefixed=args.scale_prefixed)  ### Added scale_prefixed
				
				loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
				if args.model_name == 'histoCAE' or args.model_name == 'histoCAE_latent':
					output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1, patch_size = args.target_patch_size)
				else:
					output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

				time_elapsed = time.time() - time_start
				print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

				with h5py.File(output_file_path, "r") as file:
					features = file['features'][:]
					print('features size: ', features.shape)
					print('coordinates size: ', file['coords'].shape)

				features = torch.from_numpy(features)
				bag_base, _ = os.path.splitext(bag_name)
				torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))




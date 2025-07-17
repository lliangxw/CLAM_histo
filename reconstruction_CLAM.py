import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from network import*
from histoCAE import *
from torchvision import transforms, utils
import torch.nn.functional as F
import os
import math
import time
import pytorch_ssim
from dataset import*
from tqdm import tqdm
import argparse
import openslide
from utils import*

np.set_printoptions(threshold=np.inf)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Feature Extraction')
    # parser.add_argument('--data_h5_dir', type=str, default=None)
    # parser.add_argument('--data_slide_dir', type=str, default=None)
    # parser.add_argument('--slide_ext', type=str, default= '.svs')
    # parser.add_argument('--csv_path', type=str, default=None)
    # parser.add_argument('--target_patch_size', type=int, default=224)
    # parser.add_argument('--resolution_level', type=int, default=0)
    # parser.add_argument('--scale_prefixed', type=int, default=-1)
    # args = parser.parse_args()

    model_name = 'AE_CLAM_level_0_5.pth'
    target_patch_size = 224

    device = torch.device('cpu')
    patch_size = 224
    model = CAE_Autoencoder_CLAM(patch_size)
    model.load_state_dict(torch.load('/home/local-admin/Documents/projects/CLAM/AE_CLAM_model/' + model_name, map_location=torch.device('cpu')))
    model.eval()
 
    input_folder_name = 'reconstruction_level_0_5_input'
    input_img_dir = '/home/local-admin/Documents/projects/CLAM/AE_CLAM_reconstruction/' + input_folder_name

    result_folder_name = 'reconstruction_level_0_5'
    reconst_img_dir = '/home/local-admin/Documents/projects/CLAM/AE_CLAM_reconstruction/' + result_folder_name

    create_dir(input_img_dir)
    create_dir(reconst_img_dir)

    # csv_path = args.csv_path
    # if csv_path is None:
    #     raise NotImplementedError
    
    # bags_dataset = Dataset_All_Bags(csv_path)
    # total = len(bags_dataset)

    # loader_kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}

    # wsi_infos = []

    # for bag_candidate_idx in tqdm(range(total)):
    #     slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
    #     bag_name = slide_id+'.h5'
    #     h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

	# 	# slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext) ##### Commented

	# 	### replaced by :
    #     g = os.walk(args.data_slide_dir)

    #     for dirpath, dirnames, filenames in g:
    #         for filename in filenames:
    #             if filename == slide_id+args.slide_ext:
    #                 slide_file_path = os.path.join(dirpath, filename)

    #                 wsi_infos.append((h5_file_path, slide_file_path))

    #     print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
    #     print(slide_id)

    # dataset = MultiWSISequentialDataset(wsi_infos, args.resolution_level, args.scale_prefixed, train=False)
    # dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True)

    # count = 0

    # for i, batch in enumerate(dataloader):

    #     img = batch['img']

    #     _, reconstructed = model(img, args.target_patch_size)

    #     output_tensor = torch.squeeze(reconstructed.permute(0, 2, 3, 1))  ##reconstructed

    #     output_img = output_tensor.detach().numpy()

    #     output_img = np.array(output_img)[:, :, ::-1].copy()

    #     output_img = cv2.normalize(output_img, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) ### output_img

    #     output_img = np.asarray(output_img, dtype = np.uint8)

    #     cv2.imwrite(reconst_img_dir + '/' + str(i) + '.png', output_img)

    #     count += 1

    #     if count > 5000:
    #         break

    crop_path =  '/home/local-admin/Documents/projects/MAIA-main/data_crop_0_5' 

    pic_path = make_path_bis(crop_path)

    ### Check if the images are validated
    # count_check = 0
    # for img_path in pic_path:
    #     count_check += 1
    #     if count_check%1000 == 0:
    #         print(count_check)
    #     try:
    #         img = Image.open(img_path)
    #         img.verify()  # Verify that it's a valid image
    #     except (IOError, SyntaxError) as e:
    #         print(f'Bad file: {img_path} — {e}')
    # pic_path_1, pic_path_2 = make_path_AE_pair(root)

    val_data = trainset(pic_path, train=False)
    # val_data = trainset(pic_path, train=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4) ##   

    count = 0

    for i, img in enumerate(val_dataloader):

        _, reconstructed = model(img)

        output_tensor = torch.squeeze(reconstructed.permute(0, 2, 3, 1))  ##reconstructed

        output_img = output_tensor.detach().numpy()

        output_img = np.array(output_img)[:, :, ::-1].copy()

        output_img = cv2.normalize(output_img, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) ### output_img

        output_img = np.asarray(output_img, dtype = np.uint8)


        img_tensor = torch.squeeze(img.permute(0, 2, 3, 1))

        input_img = img_tensor.detach().numpy()

        input_img = np.array(input_img)[:, :, ::-1].copy()

        input_img = cv2.normalize(input_img, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) ### output_img

        input_img = np.asarray(input_img, dtype = np.uint8)

        cv2.imwrite(input_img_dir + '/' + str(i) + '.png', input_img)

        cv2.imwrite(reconst_img_dir + '/' + str(i) + '.png', output_img)

        count += 1

        if count > 1000:
            break
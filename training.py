from dataset import *
from network import *
from histoCAE import*
import torch
from utils import*
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pytorch_ssim
from torch.utils.tensorboard import SummaryWriter

import os
import time

import gc
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=20000'

torch.set_flush_denormal(True)

loss_function_1 = torch.nn.MSELoss()  ### model_CAE_1
loss_function_2 = pytorch_ssim.SSIM()  ### model_CAE_2
loss_function_3 = nn.L1Loss() ### model_CAE_3

Loss_list = []
Accuracy_list = []

BATCH_SIZE1 = 512
BATCH_SIZE2 = 32
LEARNING_RATE = 0.001
EPOCH = 200
torch.cuda.empty_cache()

def train_model():
    model.train()

    for epoch in range(EPOCH):

        running_loss = 0

        loss_all = 0

        seconds = time.time()

        for i, img in enumerate(train_dataloader):

            # print(img.shape)

            optimizer.zero_grad()

            img = Variable(img).cuda()

            reconstructed = model(img)

            # img = torch.squeeze(img, 1)  ############ For 3D filters

            loss = loss_function_1(reconstructed, img) + (1 - loss_function_2(reconstructed, img)) + loss_function_3(reconstructed, img)

            loss.backward()

            optimizer.step()

            running_loss += float(loss)

            loss_all += float(loss)

            if (i + 1) % 10 == 0:
                print('[EPOCH = %d][iteration = %d]loss:%.8f' % (
                    epoch, i + 1, running_loss / 10))

                running_loss = 0

                torch.save(model.state_dict(), '/home.local/local-admin/Documents/data_liang/MAIA/HES/group/models/models_3_' + str(patch_size) + '/' + model_name)

                # print(time.time() - seconds)

                # seconds = time.time()

            torch.cuda.empty_cache()

            # seconds = time.time()

        print("loss of the epoch: " + str(loss_all))

        print('This Epoch takes : ' + str(time.time() - seconds) + ' seconds')
        # if loss_all > 1:
        #     loss_all = 1
        Loss_list.append(loss_all)
        #


        ##### Validation ##########
        # model.eval()
        
        # val_err = 0
        
        # for j, val_img in enumerate(val_dataloader):
        #     with torch.no_grad():
        
        #         val_img = val_img.cuda()
        #         val_reconst = model(val_img)
        
        #         # val_reconst = val_reconst.permute(0, 2, 3, 1).contiguous().view(-1, OUTPUT_CH)
        
        #         val_err += loss_function_1(val_reconst, val_img) + (1 - loss_function_2(val_reconst, val_img)) + loss_function_3(val_reconst, val_img)
        
        # # val_err = float(val_right / val_sample)
        # print('[EPOCH = %d]Accuracy(error):%.6f' % (epoch, val_err))
        # # # print(val_right, val_sample)
        # # # logger.info('[EPOCH = %d]accuracy:%.6f' % (epoch, val_acc))
        # #
        # # Accuracy_list.append(val_acc)
        #
        scheduler.step()
        #
        # model.train()


# def train_model_augmented():
#     model.train()

#     for epoch in range(EPOCH):

#         running_loss = 0

#         loss_all = 0

#         for i, (img_1, img_2) in enumerate(train_dataloader):

#             optimizer.zero_grad()

#             img_1 = Variable(img_1).cuda()
#             img_2 = Variable(img_2).cuda()

#             reconstructed_1 = model(img_1)
#             reconstructed_2 = model(img_2)

#             # reconstructed = reconstructed.permute(0, 2, 3, 1)

#             loss_1 = loss_function_1(reconstructed_1, img_1) + (1 - loss_function_2(reconstructed_1, img_1)) + loss_function_3(reconstructed_1, img_1)
#             loss_2 = loss_function_1(reconstructed_2, img_2) + (1 - loss_function_2(reconstructed_2, img_2)) + loss_function_3(reconstructed_2, img_2)
#             loss = loss_1 + loss_2

#             loss.backward()
#             optimizer.step()

#             running_loss += float(loss)

#             loss_all += float(loss)

#             if (i + 1) % 10 == 0:
#                 print('[EPOCH = %d][iteration = %d]loss:%.8f' % (
#                     epoch, i + 1, running_loss / 10))

#                 running_loss = 0

#                 torch.save(model.state_dict(), '/home/liang/data/'+ img_svs_name +'/model_augmented/' + model_name)

#         print(loss_all)
#         # if loss_all > 1:
#         #     loss_all = 1
#         Loss_list.append(loss_all)
#         #
#         # model.eval()
#         #
#         # val_sample = 0
#         # val_right = 0
#         #
#         # for j, img in enumerate(val_dataloader):
#         #     with torch.no_grad():
#         #
#         #         val_reconst = model(img)
#         #
#         #         val_reconst = val_reconst.permute(0, 2, 3, 1).contiguous().view(-1, OUTPUT_CH)
#         #
#         #         ele, right = val_des_2(val_des1, val_des1_, val_des2, val_des2_)
#         #         val_sample += ele
#         #         val_right += right
#         #
#         # val_acc = float(val_right / val_sample)
#         # print('[EPOCH = %d]accuracy:%.6f' % (epoch, val_acc))
#         # # print(val_right, val_sample)
#         # # logger.info('[EPOCH = %d]accuracy:%.6f' % (epoch, val_acc))
#         #
#         # Accuracy_list.append(val_acc)
#         #
#         # scheduler.step()
#         #
#         # model.train()

# if __name__ == "__main__":

#     img_svs_names = ['20AG05811-04_HES']
#     resolutions = [10, 20, 5]

#     # os.environ["CUDA_DEVICE_ORDER"] = "PCI_VUS_ID"
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#     for img_svs_name in img_svs_names:

#         for resolution in resolutions:

#         # resolution = 5
#             model_name = 'model_CAE_r_' + str(resolution) + '.pth'
#             # model_name_init = 'model_CAE_r_' + str(resolution) + '_init.pth'
#             # img_svs_name = '20AG05811-04_HES'
#             patch_size = 128

#             print(torch.cuda.is_available())

#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             model = CAE_Autoencoder_3()
#             # model.load_state_dict(torch.load('/home/liang/data/'+ img_svs_name +'/model_128/' + model_name_init, map_location=device))
#             model = model.to(device)

#             root = '/home/liang/data/'+ img_svs_name +'/patches_' + str(patch_size) +'_' + str(resolution)
#             # root = '/home/liang/data/'+ img_svs_name +'/patch_tensor_' + str(patch_size) +'_' + str(resolution)
#             if os.path.isdir('/home/liang/data/'+ img_svs_name +'/model_128') == False:
#                 os.mkdir('/home/liang/data/'+ img_svs_name +'/model_128')

#             # pic_path = make_path(root, 500)
#             pic_path = make_path_bis(root)

#             # train_data = trainset_augmented(pic_path) ### with rotation
#             train_data = trainset(pic_path)
#             # val_data = trainset(pic_path, train=False)
#             train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE1, shuffle=True, num_workers=12) ##
#             # val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE2, shuffle=False, num_workers=12)

#             # criterion = my_loss().cuda()
#             optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.05, 0.999))
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

#             train_model()


    # img_svs_names = ['20AG05811-04_HES']
    # resolutions = [10]
    # patch_size = 128

    # for img_svs_name in img_svs_names:

    #     for resolution in resolutions:

    #         root = '/home/liang/data/'+ img_svs_name +'/patches_' + str(patch_size) +'_' + str(resolution)
    #         pic_path = make_path_bis(root)


    #         for i in range(2560):
    #                 # img = cv2.imread(path)
    #             seconde = cv2.getTickCount()

    #             # img = Image.open(pic_path[i])
    #             img = cv2.imread(pic_path[i])

    #             seconde_2 = cv2.getTickCount()
                
    #             print((seconde_2 - seconde)/cv2.getTickFrequency())
    #             if ((seconde_2 - seconde)/cv2.getTickFrequency() < 0.01):
    #                 cv2.imwrite('/home/liang/data/'+ img_svs_name +'/patches_test/' + str(i) +'.png', img)


    #             # img = torch.from_numpy(img).float().permute(2, 0, 1)
    #             # img_tensor = img/255.0


# if __name__ == "__main__":

#     img_svs_names = ['14AG09488-06_HES']
#     resolutions = [5, 10, 20]

#     # os.environ["CUDA_DEVICE_ORDER"] = "PCI_VUS_ID"
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#     for img_svs_name in img_svs_names:

#         for resolution in resolutions:

#         # resolution = 5
#             model_name = 'model_CAE_r_' + str(resolution) + '.pth'
#             model_name_test = 'model_CAE_r_' + str(resolution) + '_test.pth'
#             # model_name_init = 'model_CAE_r_' + str(resolution) + '_init.pth'
#             # img_svs_name = '20AG05811-04_HES'
#             patch_size = 128

#             print(torch.cuda.is_available())

#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             model = CAE_Autoencoder_4()
#             # model.load_state_dict(torch.load('/home/liang/data/'+ img_svs_name +'/model_128/' + model_name, map_location=device))  #### alternative
#             model = model.to(device)

#             root = '/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name +'/patches/patches_' + str(patch_size) + '/patches_' + str(patch_size) +'_' + str(resolution)
#             # root = '/home/liang/data/'+ img_svs_name +'/patch_tensor_' + str(patch_size) +'_' + str(resolution)
#             if os.path.isdir('/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name +'/models/') == False:
#                 os.mkdir('/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name +'/models/')

#             if os.path.isdir('/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name +'/models/models_128/') == False:
#                 os.mkdir('/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name +'/models/models_128/')

#             # pic_path = make_path_bis(root)

            
#             ##-------------------- First time of reading patches ------------------------------------##

#             image_matrix = make_image_matrix(root)  ### modified method

#             image_matrix_save = np.asarray(image_matrix)
#             create_dir('/home.local/local-admin/Documents/data_liang/MAIA/HES/' + img_svs_name + '/tensors_' + str(patch_size) )
#             np.save('/home.local/local-admin/Documents/data_liang/MAIA/HES/' + img_svs_name + '/tensors_' + str(patch_size) + '/samples_' +str(resolution) + '.npy', image_matrix_save)
#             ##---------------------------------------------------------------------------------------##

#             ##----------------------For patches already saved in matrix -----------------------------##
#             # image_matrix = np.load('/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name + '/tensors_' + str(patch_size) + '/samples_' +str(resolution) + '.npy')
#             ##---------------------------------------------------------------------------------------##

#             # train_data = trainset_augmented(pic_path) ### with rotation
#             # train_data = trainset(pic_path)
#             train_data = trainset_matrix(image_matrix)  #### modified method
#             # train_data = trainset_matrix_3D(image_matrix)  #### for 3D filter
#             # val_data = trainset(pic_path, train=False)
#             train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE1, shuffle=True, num_workers=12) ##
#             # val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE2, shuffle=False, num_workers=12)

#             # criterion = my_loss().cuda()
#             optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.05, 0.999))
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#             train_model()


def train_model_CLAM_AE():

    writer = SummaryWriter(log_dir="log_loss_CLAM/AE")

    model.train()

    for epoch in range(12, EPOCH):

        running_loss = 0

        loss_all = 0

        seconds = time.time()

        for i, img in enumerate(train_dataloader):

            # print(img.shape)

            optimizer.zero_grad()

            img = Variable(img).cuda()

            _, reconstructed = model(img)

            # img = torch.squeeze(img, 1)  ############ For 3D filters

            loss = loss_function_1(reconstructed, img) + (1 - loss_function_2(reconstructed, img)) + loss_function_3(reconstructed, img)

            if epoch == 0 and i == 0:
                writer.add_scalar("Loss/train", loss.item(), 0)

            loss.backward()

            optimizer.step()

            running_loss += float(loss)

            loss_all += float(loss)

            if (i + 1) % 10 == 0:
                print('[EPOCH = %d][iteration = %d]loss:%.8f' % (
                    epoch, i + 1, running_loss / 10))

                running_loss = 0

                torch.save(model.state_dict(), '/home/local-admin/Documents/projects/CLAM/AE_CLAM_model/' + model_name)

                global_step = epoch * len(train_dataloader) + i
                writer.add_scalar("Loss/train", loss.item(), global_step)

                # print(time.time() - seconds)

                # seconds = time.time()

            torch.cuda.empty_cache()

            # seconds = time.time()

        print("loss of the epoch: " + str(loss_all))

        print('This Epoch takes : ' + str(time.time() - seconds) + ' seconds')

        torch.save(model.state_dict(), '/home/local-admin/Documents/projects/CLAM/AE_CLAM_model/' + model_name)
        # if loss_all > 1:
        #     loss_all = 1
        Loss_list.append(loss_all)
        #


        ##### Validation ##########
        # model.eval()
        
        # val_err = 0
        
        # for j, val_img in enumerate(val_dataloader):
        #     with torch.no_grad():
        
        #         val_img = val_img.cuda()
        #         val_reconst = model(val_img)
        
        #         # val_reconst = val_reconst.permute(0, 2, 3, 1).contiguous().view(-1, OUTPUT_CH)
        
        #         val_err += loss_function_1(val_reconst, val_img) + (1 - loss_function_2(val_reconst, val_img)) + loss_function_3(val_reconst, val_img)
        
        # # val_err = float(val_right / val_sample)
        # print('[EPOCH = %d]Accuracy(error):%.6f' % (epoch, val_err))
        # # # print(val_right, val_sample)
        # # # logger.info('[EPOCH = %d]accuracy:%.6f' % (epoch, val_acc))
        # #
        # # Accuracy_list.append(val_acc)
        #
        scheduler.step()
        #
        # model.train()

# if __name__ == "__main__":

#     img_svs_names = [ '12AG02477-12_HES-MSI_CC',  '14AG09488-06_HES-MSI_CHC', '15AG01973-06_HES-MSI_CHC_1_Scils', '16AG06295-12_HES-MSI_TM', '20AG02608-05_HES-MSI_TM','B0450056-12_HES-MSI_CC']

# ## All    # img_svs_names = [ '12AG02477-12_HES-MSI_CC', '13AG02766-05_HES-MSI_CC', '14AG09488-06_HES-MSI_CHC', '15AG00195-04_HES-MSI_CHC', '15AG01973-06_HES-MSI_CHC_1_Scils', '16AG05498-06_HES-MSI_CC','16AG06295-12_HES-MSI_TM', '20AG02608-05_HES-MSI_TM', '20AG05811-04_HES-MSI_TM','21AG06292-24_HES-MSI_TM', '21AG06292-25_HES-MSI_TM', 'B0450056-12_HES-MSI_CC']

#     resolutions = [10, 20, 5]

#     # os.environ["CUDA_DEVICE_ORDER"] = "PCI_VUS_ID"
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#     # for img_svs_name in img_svs_names:

#     for resolution in resolutions:

#     # resolution = 5
#         model_name = 'model_CAE_group_r_' + str(resolution) + '.pth'
#         # model_name_init = 'model_CAE_r_' + str(resolution) + '_init.pth'
#         # img_svs_name = '20AG05811-04_HES'
#         patch_size = 256

#         print(torch.cuda.is_available())

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model = CAE_Autoencoder_4()
#         # model.load_state_dict(torch.load('/home/liang/data/'+ img_svs_name +'/model_128/' + model_name, map_location=device))  #### alternative
#         model = model.to(device)

#         # root = '/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name +'/patches/patches_' + str(patch_size) + '/patches_' + str(patch_size) +'_' + str(resolution)
#         # root = '/home/liang/data/'+ img_svs_name +'/patch_tensor_' + str(patch_size) +'_' + str(resolution)
#         create_dir('/home.local/local-admin/Documents/data_liang/MAIA/HES/group')
#         create_dir('/home.local/local-admin/Documents/data_liang/MAIA/HES/group/models')
#         create_dir('/home.local/local-admin/Documents/data_liang/MAIA/HES/group/models/models_3_' + str(patch_size))

#         #---------------------- For reading images directly without saving as .npy --------------###
#         paths_all = []
#         for img_svs_name in img_svs_names:
#             paths_all.append('/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name +'/patches/patches_' + str(patch_size) + '_label' + '/patches_' + str(patch_size) +'_' + str(resolution))

#         pic_path = make_path_groups(paths_all)
        
#         ##-------------------- First time of reading patches ------------------------------------##

#         # image_matrix = make_image_matrix(root)  ### modified method

#         # image_matrix_save = np.asarray(image_matrix)
#         # create_dir('/home.local/local-admin/Documents/data_liang/MAIA/HES/' + img_svs_name + '/tensors_' + str(patch_size) )
#         # np.save('/home.local/local-admin/Documents/data_liang/MAIA/HES/' + img_svs_name + '/tensors_' + str(patch_size) + '/samples_' +str(resolution) + '.npy', image_matrix_save)
#         ##---------------------------------------------------------------------------------------##

#         ##----------------------For patches already saved in matrix -----------------------------##
#         # image_matrix = np.load('/home.local/local-admin/Documents/data_liang/MAIA/HES/'+ img_svs_name + '/tensors_' + str(patch_size) + '/samples_' +str(resolution) + '.npy')
#         ##---------------------------------------------------------------------------------------##

#         # train_data = trainset_augmented(pic_path) ### with rotation
#         train_data = trainset(pic_path)  ##### for reading images directly
#         # train_data = trainset_matrix(image_matrix)  #### modified method
#         # train_data = trainset_matrix_3D(image_matrix)  #### for 3D filter
#         # val_data = trainset(pic_path, train=False)
#         train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE1, shuffle=True, num_workers=12) ##
#         # val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE2, shuffle=False, num_workers=12)

#         # criterion = my_loss().cuda()
#         optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.05, 0.999))
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

#         train_model()


if __name__ == "__main__":


# resolution = 5
    model_name_0 = 'AE_CLAM_level_0_5.pth'
    model_name = 'AE_CLAM_level_0_5.pth'
    # model_name_init = 'model_CAE_r_' + str(resolution) + '_init.pth'
    # img_svs_name = '20AG05811-04_HES'
    patch_size = 224

    print(torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CAE_Autoencoder_CLAM(patch_size)
    model.load_state_dict(torch.load('/home.local/local-admin/Documents/projects/CLAM/AE_CLAM_model/' + model_name_0, map_location=device))  #### alternative
    model = model.to(device)

    data_crop_dir = '/home/local-admin/Documents/projects/MAIA-main/data_crop_0_5' 

    pic_path = make_path_bis(data_crop_dir)

    # train_data = trainset_augmented(pic_path) ### with rotation
    train_data = trainset(pic_path)  ##### for reading images directly
    # train_data = trainset_matrix(image_matrix)  #### modified method
    # train_data = trainset_matrix_3D(image_matrix)  #### for 3D filter
    # val_data = trainset(pic_path, train=False)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE1, shuffle=True, num_workers=12) ##
    # val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE2, shuffle=False, num_workers=12)

    # criterion = my_loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.05, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    train_model_CLAM_AE()
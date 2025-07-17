import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torchinfo import summary
import math


# class CAE_Autoencoder_visapp(nn.Module):

#     def __init__(self, patch_size):
#         super(CAE_Autoencoder_visapp, self).__init__()

#         self.encoder = nn.Sequential(
#             ConvBnRelu(3, 10, kernel_size=3, padding = 'same'),
#             nn.MaxPool2d(2, stride=2, padding = 0),
#             ConvBnRelu(10, 20, kernel_size=3, padding = 'same'),
#             nn.MaxPool2d(2, stride=2, padding = 0),
#             nn.Flatten(),
#             DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 2))*20), 500, 0.0)      
#         )

#         self.resize_layer = nn.Sequential(
#             DenseBnRelu(500, int(patch_size*patch_size/(pow(2*2, 2))*20), 0.0)
#         )

#         self.decoder = nn.Sequential(
            
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             ConvBnRelu(20, 10, kernel_size=3, padding = 'same'),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             ConvBnRelu(10, 3, kernel_size=3, padding = 'same'),

#             nn.Conv2d(3, 3, kernel_size=3, padding='same'),
#             nn.BatchNorm2d(3),
#             nn.Sigmoid()
#         )

#     def forward(self, x, patch_size):

#         encoded = self.encoder(x)
#         encoded_1 = self.resize_layer(encoded)
#         encoded_1 = torch.reshape(encoded_1, (-1, 20, int(patch_size/pow(2, 2)), int(patch_size/pow(2, 2))))
#         decoded = self.decoder(encoded_1)
#         # decoded = self.decoder(encoded)

#         return encoded, decoded


class CAE_Autoencoder_visapp(nn.Module):

    def __init__(self, patch_size):
        super(CAE_Autoencoder_visapp, self).__init__()

        self.encoder = nn.Sequential(
            # ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(3, 16, kernel_size=3, stride=2),
            # ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 32, kernel_size=3, stride=2),
            # ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 64, kernel_size=3, stride=2),
            nn.Flatten(),
            DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 3))*64), 1024, 0.0)   
               
        )

        self.resize_layer = nn.Sequential(
            DenseBnRelu(1024, int(patch_size*patch_size/(pow(2*2, 3))*64), 0.0)
        )

        self.decoder = nn.Sequential(
            
            # ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 3, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(3, 3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x, patch_size):

        encoded = self.encoder(x)
        encoded = F.normalize(encoded, p=2.0, dim = 1)
        encoded_1 = self.resize_layer(encoded)
        encoded_1 = torch.reshape(encoded_1, (-1, 64, int(patch_size/pow(2, 3)), int(patch_size/pow(2, 3))))
        decoded = self.decoder(encoded_1)
        # decoded = self.decoder(encoded)

        return encoded, decoded



# class CAE_Autoencoder_visapp(nn.Module):

#     def __init__(self, patch_size):
#         super(CAE_Autoencoder_visapp, self).__init__()

#         self.encoder = nn.Sequential(
#             ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
#             nn.MaxPool2d(2, stride=2, padding = 0),
#             ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
#             nn.MaxPool2d(2, stride=2, padding = 0),
#             nn.Flatten(),
#             DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 2))*32), 50, 0.0)
            
#         )

#         self.resize_layer = nn.Sequential(
#             DenseBnRelu(50, int(patch_size*patch_size/(pow(2*2, 2))*32), 0.0)
#         )

#         self.decoder = nn.Sequential(
            
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             ConvBnRelu(16, 3, kernel_size=3, padding = 'same'),
#             nn.Conv2d(3, 3, kernel_size=3, padding='same'),
#             nn.BatchNorm2d(3),
#             nn.Sigmoid()
#         )

#     def forward(self, x, patch_size):

#         encoded = self.encoder(x)
#         encoded_1 = self.resize_layer(encoded)
#         encoded_1 = torch.reshape(encoded_1, (-1, 32, int(patch_size/pow(2, 2)), int(patch_size/pow(2, 2))))
#         decoded = self.decoder(encoded_1)

#         return encoded, decoded

class CAE_Autoencoder(nn.Module):

    def __init__(self):
        super(CAE_Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid(),
            nn.Conv2d(16, 3, kernel_size=3, padding='same')
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class CAE_Autoencoder_2(nn.Module):

    def __init__(self):
        super(CAE_Autoencoder_2, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid(),
            nn.Conv2d(16, 3, kernel_size=3, padding='same')
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

class CAE_Autoencoder_3(nn.Module):

    def __init__(self):
        super(CAE_Autoencoder_3, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 128, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(128, 128, kernel_size=3, stride=2)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(128, 128, kernel_size=3, padding = 'same'),
            ConvBnRelu(128, 128, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(128, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid(),
            nn.Conv2d(16, 3, kernel_size=3, padding='same')
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

class CAE_Autoencoder_4(nn.Module):

    def __init__(self):
        super(CAE_Autoencoder_4, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid(),
            nn.Conv2d(16, 3, kernel_size=3, padding='same')
            
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

    def forward(self, x):

        encoded = self.encoder(x)

        return encoded

class fc(nn.Module):

    def __init__(self, patch_size, num_class):
        super(fc, self).__init__()

        self.intermediate = nn.Sequential(
            ConvBnRelu_stride(64, 128, kernel_size=3, stride=2),
        )

        self.output = nn.Sequential(
            nn.Flatten(),
            DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 5))*128), 4096, 0.7),
            DenseBnRelu(4096, 4096, 0.7),
            DenseBnRelu(4096, 256, 0.7),
            nn.Linear(256, num_class)
        )

    def forward(self, x):

        output = self.intermediate(x)
        output = self.output(output)

        return output

class CAE_Segmentation(nn.Module):

    def __init__(self, patch_size, num_class):
        super(CAE_Segmentation, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.segmentation = fc(patch_size, num_class)

    def forward(self, x):
    
        x = self.encoder(x)
        output = self.segmentation(x)

        return output

class CAE_Segmentation_3_R(nn.Module):

    def __init__(self, patch_size, num_class):
        super(CAE_Segmentation_3_R, self).__init__()

        self.encoder_5 = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.encoder_10 = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.encoder_20 = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        for param in self.encoder_5.parameters():
            param.requires_grad = False

        for param in self.encoder_10.parameters():
            param.requires_grad = False

        for param in self.encoder_20.parameters():
            param.requires_grad = False

        self.intermediate_1 = nn.Sequential(
            ConvBnRelu_stride(64, 128, kernel_size=3, stride=2),
            ConvBnRelu_stride(128, 256, kernel_size=3, stride=2),
            ConvBnRelu_stride(256, 512, kernel_size=3, stride=2),
            nn.Flatten()
        )

        self.intermediate_2 = nn.Sequential(
            ConvBnRelu_stride(64, 128, kernel_size=3, stride=2),
            ConvBnRelu_stride(128, 256, kernel_size=3, stride=2),
            ConvBnRelu_stride(256, 512, kernel_size=3, stride=2),
            nn.Flatten()
        )

        self.intermediate_3 = nn.Sequential(
            ConvBnRelu_stride(64, 128, kernel_size=3, stride=2),
            ConvBnRelu_stride(128, 256, kernel_size=3, stride=2),
            ConvBnRelu_stride(256, 512, kernel_size=3, stride=2),
            nn.Flatten()
        )

        self.fc_part = nn.Sequential(
            DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 7))*512)*3, 512, 0.5),
            DenseBnRelu(512, 512, 0.5),
            nn.Linear(512, num_class)
        )

    def forward(self, x_5, x_10, x_20):

        x_5 = self.encoder_5(x_5)
        x_10 = self.encoder_10(x_10)
        x_20 = self.encoder_20(x_20)

        x_5 = self.intermediate_1(x_5)
        x_10 = self.intermediate_2(x_10)
        x_20 = self.intermediate_3(x_20)

        x_full = torch.cat((x_5, x_10, x_20), -1)

        output = self.fc_part(x_full)

        return output

# class CAE_Siamese(nn.Module):

#     def __init__(self, patch_size):
#         super(CAE_Siamese, self).__init__()

#         self.encoder_10 = nn.Sequential(
#             ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
#             ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
#             ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
#             ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
#         )

#         # for param in self.encoder_5.parameters():
#         #     param.requires_grad = False

#         # for param in self.encoder_10.parameters():
#         #     param.requires_grad = False

#         # for param in self.encoder_20.parameters():
#         #     param.requires_grad = False

#         # self.intermediate_1 = nn.Sequential(
#         #     nn.Flatten()
#         # )

#         self.decoder_10 = nn.Sequential(
#             ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
#             ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
#             ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
#             ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
#             ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Sigmoid(),
#             nn.Conv2d(16, 3, kernel_size=3, padding='same')
#         )

#         self.intermediate_2 = nn.Sequential(
#             # ConvBnRelu_stride(64, 128, kernel_size=3, stride=2),
#             # ConvBnRelu_stride(128, 256, kernel_size=3, stride=2),
#             # ConvBnRelu_stride(256, 512, kernel_size=3, stride=2),
#             nn.Flatten()
#         )

#         # self.intermediate_3 = nn.Sequential(
#         #     nn.Flatten()
#         # )

#         self.fc_part = nn.Sequential(
#             DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 4))*64), 64, 0.0),  ## DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 7))*512), 512, 0.5),
#             DenseBnRelu(64, 128, 0.0)    ####DenseBnRelu(512, 512, 0.5)
#             # nn.Linear(512, num_class)
#         )

#     def forward(self, x_10):

#         # x_5 = self.encoder_5(x_5)
#         encoded = self.encoder_10(x_10)
#         # x_20 = self.encoder_20(x_20)

#         # x_5 = self.intermediate_1(x_5)
#         x_10 = self.intermediate_2(encoded)
#         # x_20 = self.intermediate_3(x_20)

#         # x_full = torch.cat((x_5, x_10, x_20), -1)

#         # output_1 = functional.normalize(x_10, p=2.0, dim = 0)

#         output = self.fc_part(x_10)
#         # print(output)
#         output = F.normalize(output, p=2.0, dim = 1)
#         # print(output)

#         decoded = self.decoder_10(encoded)

#         return output, decoded



class CAE_Siamese(nn.Module):

    def __init__(self, patch_size):
        super(CAE_Siamese, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 10, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0),
            ConvBnRelu(10, 20, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0),
            nn.Flatten(),
            DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 2))*20), 50, 0.0)            
        )

        for param in self.encoder[0: 5].parameters():
            param.requires_grad = False


        self.resize_layer = nn.Sequential(
            DenseBnRelu(50, int(patch_size*patch_size/(pow(2*2, 2))*20), 0.0)
        )

        self.decoder = nn.Sequential(
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(20, 10, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(10, 3, kernel_size=3, padding = 'same'),
            nn.Conv2d(3, 3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x, patch_size):

        encoded = self.encoder(x)
        encoded_1 = self.resize_layer(encoded)
        # print(encoded.shape)
        encoded_2 = torch.reshape(encoded_1, (-1, 20, int(patch_size/pow(2, 2)), int(patch_size/pow(2, 2))))
        decoded = self.decoder(encoded_2)

        output = F.normalize(encoded, p=2.0, dim = 1)

        return output, decoded


class CAE_Autoencoder_4_output(nn.Module):

    def __init__(self):
        super(CAE_Autoencoder_4_output, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):

        list_output = []

        encoded_1 = ConvBnRelu(3, 16, kernel_size=3, padding = 'same')(x)
        encoded_2 = ConvBnRelu_stride(16, 16, kernel_size=3, stride=2)(encoded_1)

        encoded_3 = ConvBnRelu(16, 32, kernel_size=3, padding = 'same')(encoded_2)
        encoded_4 = ConvBnRelu_stride(32, 32, kernel_size=3, stride=2)(encoded_3)

        encoded_5 = ConvBnRelu(32, 64, kernel_size=3, padding = 'same')(encoded_4)
        encoded_6 = ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)(encoded_5)

        encoded_7 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(encoded_6)
        encoded_8 = ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)(encoded_7)

        decoded_1 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(encoded_8)
        decoded_2 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(decoded_1)
        decoded_2 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_2)

        decoded_3 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(decoded_2)
        decoded_4 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(decoded_3)
        decoded_4 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_4)

        decoded_5 = ConvBnRelu(64, 32, kernel_size=3, padding = 'same')(decoded_4)
        decoded_6 = ConvBnRelu(32, 32, kernel_size=3, padding = 'same')(decoded_5)
        decoded_6 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_6)

        decoded_7 = ConvBnRelu(32, 16, kernel_size=3, padding = 'same')(decoded_6)
        decoded_8 = ConvBnRelu(16, 16, kernel_size=3, padding = 'same')(decoded_7)
        decoded_8 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_8)

        list_output = [encoded_1, encoded_2, encoded_3, encoded_4, encoded_5, encoded_6, encoded_7, encoded_8, decoded_1, decoded_2, decoded_3, decoded_4, decoded_5, decoded_6, decoded_7, decoded_8]

        return list_output


class CAE_Autoencoder_4_output_test(nn.Module):

    def __init__(self):
        super(CAE_Autoencoder_4_output, self).__init__()

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):

        list_output = []

        encoded_1 = ConvBnRelu(3, 16, kernel_size=3, padding = 'same')(x)
        encoded_2 = ConvBnRelu_stride(16, 16, kernel_size=3, stride=2)(encoded_1)

        encoded_3 = ConvBnRelu(16, 32, kernel_size=3, padding = 'same')(encoded_2)
        encoded_4 = ConvBnRelu_stride(32, 32, kernel_size=3, stride=2)(encoded_3)

        encoded_5 = ConvBnRelu(32, 64, kernel_size=3, padding = 'same')(encoded_4)
        encoded_6 = ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)(encoded_5)

        # encoded_7 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(encoded_6)
        # encoded_8 = ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)(encoded_7)

        # decoded_1 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(encoded_8)
        # decoded_2 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(decoded_1)
        # decoded_2 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_2)

        # decoded_3 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(decoded_2)
        # decoded_4 = ConvBnRelu(64, 64, kernel_size=3, padding = 'same')(decoded_3)
        # decoded_4 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_4)

        # decoded_5 = ConvBnRelu(64, 32, kernel_size=3, padding = 'same')(decoded_4)
        # decoded_6 = ConvBnRelu(32, 32, kernel_size=3, padding = 'same')(decoded_5)
        # decoded_6 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_6)

        # decoded_7 = ConvBnRelu(32, 16, kernel_size=3, padding = 'same')(decoded_6)
        # decoded_8 = ConvBnRelu(16, 16, kernel_size=3, padding = 'same')(decoded_7)
        # decoded_8 = nn.Upsample(scale_factor=2, mode='bilinear')(decoded_8)

        return encoded_6

class CAE_Autoencoder_5(nn.Module):

    def __init__(self):
        super(CAE_Autoencoder_5, self).__init__()

        self.encoder_3D = ConvBnRelu_3D(1, 16, kernel_size=3, padding = (0, 1, 1))

        self.encoder = nn.Sequential(
            # ConvBnRelu_3D(1, 16, kernel_size=3, padding = (0, 1, 1)),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder_3D(x)
        x = torch.squeeze(x, 2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


######## For CLAM, the bottleneck layer is flattened

class CAE_Autoencoder_CLAM(nn.Module):

    def __init__(self,patch_size):
        super(CAE_Autoencoder_CLAM, self).__init__()

        self.patch_size = patch_size

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            nn.Flatten(),
            DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 4))*64), 1024, 0.0),
        )

        self.resize_layer = nn.Sequential(
            DenseBnRelu(1024, int(patch_size*patch_size/(pow(2*2, 4))*64), 0.0)
        )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        encoded = F.normalize(encoded, p=2.0, dim = 1)
        encoded_1 = self.resize_layer(encoded)
        encoded_1 = torch.reshape(encoded_1, (-1, 64, int(self.patch_size/pow(2, 4)), int(self.patch_size/pow(2, 4))))
        decoded = self.decoder(encoded_1)

        return encoded, decoded



class CAE_Autoencoder_CLAM_no_flatten(nn.Module):

    def __init__(self,patch_size):
        super(CAE_Autoencoder_CLAM_no_flatten, self).__init__()

        self.patch_size = patch_size

        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            # nn.Flatten(),
            # DenseBnRelu(int(patch_size*patch_size/(pow(2*2, 4))*64), 1024, 0.0),
        )

        # self.resize_layer = nn.Sequential(
        #     DenseBnRelu(1024, int(patch_size*patch_size/(pow(2*2, 4))*64), 0.0)
        # )

        self.decoder = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(64, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu(32, 32, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(32, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 3, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        # encoded = F.normalize(encoded, p=2.0, dim = 1)
        # encoded_1 = self.resize_layer(encoded)
        # encoded_1 = torch.reshape(encoded_1, (-1, 64, int(self.patch_size/pow(2, 4)), int(self.patch_size/pow(2, 4))))
        decoded = self.decoder(encoded)

        return encoded, decoded

###############

# class ConvBnRelu(nn.Module):

#     def __init__(self, input_ch, output_ch, kernel_size, padding=0, stride = 1):
#         super(ConvBnRelu, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(output_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         x = self.conv(x)
#         return x

class ConvBnRelu(nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, padding=0, stride = 1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_ch)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class DenseBnRelu(nn.Module):

    def __init__(self, input_ch, output_ch, drop):
        super(DenseBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(input_ch, output_ch),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_ch),
            nn.Dropout(p=drop)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class ConvBnRelu_3D(nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, padding=0, stride = 1):
        super(ConvBnRelu_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding),   #### Conv3d
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(output_ch)  ##### BatchNorm3d
        )

    def forward(self,x):
        x = self.conv(x)
        return x

# class ConvBnRelu_stride(nn.Module):    ##### resolved by https://github.com/pytorch/pytorch/issues/67551, to support models that utilize TensorFlow's same padding.

#     def __init__(self, input_ch, output_ch, kernel_size, stride):
#         super(ConvBnRelu_stride, self).__init__()
#         self.conv = nn.Sequential(
#             Conv2dSame(input_ch, output_ch, kernel_size, stride, groups=1, bias=True),
#             # nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(output_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         x = self.conv(x)
#         return x

class ConvBnRelu_stride(nn.Module):    ##### resolved by https://github.com/pytorch/pytorch/issues/67551, to support models that utilize TensorFlow's same padding.

    def __init__(self, input_ch, output_ch, kernel_size, stride):
        super(ConvBnRelu_stride, self).__init__()
        self.conv = nn.Sequential(
            Conv2dSame(input_ch, output_ch, kernel_size, stride, groups=1, bias=True),
            # nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_ch)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.encoder = nn.Sequential(
        #     ConvBnRelu(3, 8, kernel_size=3, padding = 'same'),
        #     ConvBnRelu_stride(8, 8, kernel_size=3, stride=2),
        #     ConvBnRelu(8, 8, kernel_size=3, padding = 'same'),
        #     ConvBnRelu_stride(8, 8, kernel_size=3, stride=2),
        #     ConvBnRelu(8, 16, kernel_size=3, padding = 'same'),
        #     ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
        #     ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
        #     ConvBnRelu_stride(32, 32, kernel_size=3, stride=2)
        # )

            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            # ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            # ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )


        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.conv3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0 )
        # self.conv4 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        # self.conv5 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# class MyNet(nn.Module):
#     def __init__(self):
#         super(MyNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1 )
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.ModuleList()
#         self.pooling = nn.ModuleList()
#         self.bn2 = nn.ModuleList()
#         for i in range(4):
#             self.conv2.append( nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1 ) )
#             self.pooling.append(ConvBnRelu_stride(16, 16, kernel_size=3, stride=2))
#             self.bn2.append( nn.BatchNorm2d(16) )
#         self.conv3 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0 )
#         self.bn3 = nn.BatchNorm2d(16)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu( x )
#         x = self.bn1(x)
#         for i in range(4):
#             x = self.conv2[i](x)
#             x = self.pooling[i](x)
#             x = self.bn2[i](x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         return x


class MyNet_AE(nn.Module):
    def __init__(self):
        super(MyNet_AE, self).__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(3, 10, kernel_size=3, padding = 'same'),
            # ConvBnRelu(10, 10, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0),
            ConvBnRelu(10, 20, kernel_size=3, padding = 'same'),
            # ConvBnRelu(20, 20, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0),
            ConvBnRelu(20, 40, kernel_size=3, padding = 'same'),
            # ConvBnRelu(20, 20, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0),
            ConvBnRelu(40, 40, kernel_size=3, padding = 'same'),
            # ConvBnRelu(20, 20, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0)
        )

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.conv3 = nn.Conv2d(40, 16, kernel_size=1, stride=1, padding=0 )
        # self.conv4 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        # self.conv5 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class MyNet_AE_2(nn.Module):
    def __init__(self):
        super(MyNet_AE_2, self).__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(3, 10, kernel_size=3, padding = 'same'),
            # ConvBnRelu(10, 10, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0),
            # ConvBnRelu(10, 20, kernel_size=3, padding = 'same'),
            ConvBnRelu(10, 20, kernel_size=3, padding = 'same'),
            nn.MaxPool2d(2, stride=2, padding = 0)
        )

        # for param in self.encoder.parameters():
            # param.requires_grad = False

        self.conv3 = nn.Conv2d(20, 8, kernel_size=1, stride=1, padding=0 )
        # self.conv4 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        # self.conv5 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class MyNet_2(nn.Module):
    def __init__(self):
        super(MyNet_2, self).__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2),
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.conv3 = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0 )
        # self.conv4 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        # self.conv5 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class MyNet_bis(nn.Module):
    def __init__(self,input_dim):
        super(MyNet_Output, self).__init__()
        self.encoder_1 = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2)
        )

        self.encoder_2 = nn.Sequential(
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2)
        )

        self.encoder_3 = nn.Sequential(
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.encoder_4 = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0 )
        # self.conv4 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        # self.conv5 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )

        self.bn3_1 = nn.BatchNorm2d(64)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.bn3_3 = nn.BatchNorm2d(64)
        self.bn3_4 = nn.BatchNorm2d(64)

    def forward(self, x):

        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)

        x = self.conv3(x)

        x_1 = x[:, :,  0: int(x.shape[2]/2), 0: int(x.shape[3]/2)]
        x_2 = x[:, :, int(x.shape[2]/2): x.shape[2], 0: int(x.shape[3]/2)]
        x_3 = x[:, :, 0: int(x.shape[2]/2), int(x.shape[3]/2): x.shape[3]]
        x_4 = x[:, :, int(x.shape[2]/2): x.shape[2], int(x.shape[3]/2): x.shape[2]]

        x_1 = self.bn3_1(x_1)
        x_2 = self.bn3_2(x_2)
        x_3 = self.bn3_3(x_3)
        x_4 = self.bn3_4(x_4)

        x[:, :, 0: int(x.shape[2]/2), 0: int(x.shape[3]/2)] = x_1
        x[:, :, int(x.shape[2]/2): x.shape[2], 0: int(x.shape[3]/2)] = x_2
        x[:, :, 0: int(x.shape[2]/2), int(x.shape[3]/2): x.shape[3]] = x_3
        x[:, :, int(x.shape[2]/2): x.shape[2], int(x.shape[3]/2): x.shape[3]] = x_4

        return x


class MyNet_Output(nn.Module):
    def __init__(self,input_dim):
        super(MyNet_Output, self).__init__()
        self.encoder_1 = nn.Sequential(
            ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(16, 16, kernel_size=3, stride=2)
        )

        self.encoder_2 = nn.Sequential(
            ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(32, 32, kernel_size=3, stride=2)
        )

        self.encoder_3 = nn.Sequential(
            ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        self.encoder_4 = nn.Sequential(
            ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
            ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
        )

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.conv3 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0 )
        # self.conv4 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        # self.conv5 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )

        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):

        list_output = []

        x = self.encoder_1(x)
        list_output.append(x)

        x = self.encoder_2(x)
        list_output.append(x)

        x = self.encoder_3(x)
        list_output.append(x)

        x = self.encoder_4(x)
        list_output.append(x)

        x = self.conv3(x)
        list_output.append(x)

        x = self.bn3(x)
        list_output.append(x)
        return x, list_output

# class MyNet_Output(nn.Module):
#     def __init__(self,input_dim):
#         super(MyNet_Output, self).__init__()
#         self.encoder_1 = nn.Sequential(
#             ConvBnRelu(3, 16, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(16, 16, kernel_size=3, stride=2)
#         )

#         self.encoder_2 = nn.Sequential(
#             ConvBnRelu(16, 32, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(32, 32, kernel_size=3, stride=2)
#         )

#         self.encoder_3 = nn.Sequential(
#             ConvBnRelu(32, 64, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
#         )

#         self.encoder_4 = nn.Sequential(
#             ConvBnRelu(64, 64, kernel_size=3, padding = 'same'),
#             ConvBnRelu_stride(64, 64, kernel_size=3, stride=2)
#         )

#         # for param in self.encoder.parameters():
#         #     param.requires_grad = False

#         self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0 )
#         # self.conv4 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
#         # self.conv5 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )

#         self.bn3_1 = nn.BatchNorm2d(64)
#         self.bn3_2 = nn.BatchNorm2d(64)
#         self.bn3_3 = nn.BatchNorm2d(64)
#         self.bn3_4 = nn.BatchNorm2d(64)

#     def forward(self, x):

#         list_output = []
#         x = self.encoder_1(x)
#         list_output.append(x)
#         x = self.encoder_2(x)
#         list_output.append(x)
#         x = self.encoder_3(x)
#         list_output.append(x)
#         x = self.encoder_4(x)

#         list_output.append(x)
#         x = self.conv3(x)

#         x_1 = x[:, :,  0: int(x.shape[2]/2), 0: int(x.shape[3]/2)]
#         print(x_1.shape)
#         x_2 = x[:, :, int(x.shape[2]/2): x.shape[2], 0: int(x.shape[3]/2)]
#         x_3 = x[:, :, 0: int(x.shape[2]/2), int(x.shape[3]/2): x.shape[3]]
#         x_4 = x[:, :, int(x.shape[2]/2): x.shape[2], int(x.shape[3]/2): x.shape[2]]

#         list_output.append(x)
#         x_1 = self.bn3_1(x_1)
#         x_2 = self.bn3_2(x_2)
#         x_3 = self.bn3_3(x_3)
#         x_4 = self.bn3_4(x_4)

#         x[:, :, 0: int(x.shape[2]/2), 0: int(x.shape[3]/2)] = x_1
#         x[:, :, int(x.shape[2]/2): x.shape[2], 0: int(x.shape[3]/2)] = x_2
#         x[:, :, 0: int(x.shape[2]/2), int(x.shape[3]/2): x.shape[3]] = x_3
#         x[:, :, int(x.shape[2]/2): x.shape[2], int(x.shape[3]/2): x.shape[3]] = x_4

#         list_output.append(x)
#         return x, list_output

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNet()
    # model = CAE_Autoencoder_4()
    # model = model.to(device)
    summary(model, (1, 3, 5120, 5120))
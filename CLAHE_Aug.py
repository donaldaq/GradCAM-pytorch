import imgaug as ia
from imgaug import augmenters as iaa

import numpy as np
import imageio
import os
import cv2
import time


def newaug_rot(imgname):


    rotatenumber = 0
    opt = 7     
    img = imgname #imageio.imread(imgname) #read you image
    images = np.array(
        [img for _ in range(2)], dtype=np.uint8)  # 32 means creat 32 enhanced images using following methods.

    if opt == 1:
        seq = iaa.Sequential(
            [
                iaa.Affine(
                scale={
                    "x": (0.9, 0.9),
                    "y": (0.9, 0.9)
                },
                translate_percent={
                    #"x": (-0.0, 0.2)
                    #"y": (-0.10, 0.00) # minus is up, plus is down
                }
                #rotate=(rotatenumber)
                )
                
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            ],
            random_order=True)  # apply augmenters in random order
    elif opt == 2:
        seq = iaa.Sequential(
            [
                iaa.Fliplr(1),  #horizontal
            
            ],
            random_order=True)  # apply augmenters in random order
    elif opt == 3:
        seq = iaa.Sequential(
            [
                iaa.Flipud(1), #vertical
            ],
            random_order=True)  # apply augmenters in random order
    elif opt == 4:
        seq = iaa.Sequential(
            [
                iaa.Affine(
                scale={
                    "x": (1.05, 1.05),
                    "y": (1.05, 1.05)
                },
                # translate_percent={
                #     #"x": (-0.0, 0.2)
                #     "y": (-0.10, 0.00) # minus is up, plus is down
                # }
                #rotate=(rotatenumber)
                )
                
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            ],
            random_order=True)  # apply augmenters in random order
    elif opt == 5:
        seq = iaa.Sequential(
            [
                iaa.Affine(
                # scale={
                #     "x": (1.1, 1.1),
                #     "y": (1.1, 1.1)
                # },
                translate_percent={
                    #"x": (-0.0, 0.2)
                    "y": (-0.10, 0.00) # minus is up, plus is down
                }
                #rotate=(rotatenumber)
                )
                
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            ],
            random_order=True)  # apply augmenters in random order
    elif opt == 6:
        seq = iaa.Sequential(
            [
                iaa.Affine(
                # scale={
                #     "x": (1.1, 1.1),
                #     "y": (1.1, 1.1)
                # },
                # translate_percent={
                #     #"x": (-0.0, 0.2)
                #     "y": (-0.10, 0.00) # minus is up, plus is down
                # }
                rotate=(rotatenumber)
                )
                
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            ],
            random_order=True)
    elif opt == 7: 
        # seq = iaa.CLAHE(
        #         clip_limit=(2),
        #         tile_grid_size_px=(8,8)
        #     )
        seq = iaa.AllChannelsCLAHE(clip_limit=(2), per_channel=True)
    images_aug = seq.augment_images(images)


    return images_aug[0]

    # for i in range(1):
    #     img = os.path.basename(imgname)
    #     img = os.path.splitext(img)[0]
    #     #imageio.imwrite(str(i)+"new_rot{}_{}".format(rotatenumber,img), images_aug[i])  #backup code
    #     if opt == 1:
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}_rot{}.jpg".format(upperfolder,folderName,img,rotatenumber), images_aug[i])  #write all changed images
    #         print(img)
    #         print(folderName)
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}/{}_zoomout10{}".format(upperfolder,folderName,'ta',img, ext), images_aug[i])  #write all changed images
    #         print("zoomout processing...")
    #     elif opt == 2:
    #         print(img)
    #         print('current folder name: ',folderName)
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}_horizontal{}".format(upperfolder,folderName,img, ext), images_aug[i])  #write all changed images
    #         print("horizontal processing...")
    #     elif opt == 3:
    #         print(img)
    #         print('current folder name: ',folderName)
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}_vertical{}".format(upperfolder,folderName,img, ext), images_aug[i])  #write all changed images
    #         print("vertical processing...")
    #     elif opt == 4:
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}_rot{}.jpg".format(upperfolder,folderName,img,rotatenumber), images_aug[i])  #write all changed images
    #         print(img)
    #         print('current folder name: ',folderName)
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}/{}_zoomin5{}".format(upperfolder,folderName,'ta',img, ext), images_aug[i])  #write all changed images
    #         print("zoomin processing...")
    #     elif opt == 5:
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}_rot{}.jpg".format(upperfolder,folderName,img,rotatenumber), images_aug[i])  #write all changed images
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}/{}_axisup{}".format('ta',img, ext), images_aug[i])  #write all changed images
    #         print("axisup processing...")
    #     elif opt == 6:
    #         print(img)
    #         print('current folder name: ',folderName)
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}/{}_rot{}{}".format(upperfolder,folderName,'ta',img, rotatenumber, ext), images_aug[i])  #write all changed images
    #         print("rotation processing...")
    #     elif opt == 7:
    #         print(img)
    #         print('current folder name: ',folderName)
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}{}".format(upperfolder,folderName,img, ext), images_aug[i])  #write all changed images
    #         print("CLAHE processing...")
    #     elif opt == 8:
    #         print(img)
    #         print('current folder name: ',folderName)
    #         #imageio.imwrite("/home/mlm08/ml/data/{}{}/{}_rot{}{}".format(upperfolder,folderName,img, rotatenumber, ext), images_aug[i])  #write all changed images
    #         print("2times rotation processing...")

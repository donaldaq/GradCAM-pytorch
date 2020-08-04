from __future__ import print_function

import copy
import os
import sys

import cv2
import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
import torch, gc
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import models
import time

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def get_class_name(c):
    labels = np.loadtxt('stuff/colon.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

def save_gradcam(filename, gcam, raw_image, prob, prediction,paper_cmap=False):
    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    gcam = gcam.cpu().numpy()
    
    cmap = light_jet(gcam)[..., :3] *  255.0 
    #cmap = cm.jet_r(gcam)[..., :3] * 255.0
    
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    
    #original style
    #cv2.imwrite(filename, np.uint8(gcam))

    #prediction and probability
    plt.imshow(np.uint8(gcam))#[:,:,::-1]
    plt.title('{}: {:.1f}%'.format(prediction, prob))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename,bbox_inces='tight',pad_inches=0,dpi=100)


model_name = sys.argv[1]
model_type = sys.argv[2]
#label_folder = '1'
#image_dir = './test/{}'.format(label_folder)
test_dir = './test/test/'

for (path, dir, files) in os.walk(test_dir):
    #print('check: ',dir)
    for i in dir:
        print('check:', i) 
        image_dir = test_dir + i
        label_folder = i
        output_dir = './CAM/' + model_name.split('.')[0] + '/' + i
        model_dir = './Model'
        model_path = os.path.join(model_dir, model_name)

        device_id = int(sys.argv[3])
        torch.cuda.set_device(device_id)

        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except OSError:
            print("Error: Creating directory. " )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Model
def model_load():
    model = models.load(model_type)

    #model = torch.load(model_path)
    #model.load_state_dict(torch.load(model_path))

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    #model = model.cuda()
        
    model.eval()
    model.cuda()

    for param in model.parameters():
        param.requires_grad = True

    gcam = GradCAM(model = model)
    del state_dict

    return model, gcam


# The four residual layers
target_layers = [models.load_layer(model_type)]

def input_image(model, gcam):
    #model, gcam = model_load()
    for (path, dir, files) in os.walk(test_dir):
        for i in dir:
            image_dir = path+i


            print('foldername', image_dir)
            print('path', path)
            print('files',files)

            for image_name in os.listdir(image_dir):
                #model, gcam = model_load()    
                image_path = os.path.join(image_dir, image_name)

                #raw_image = cv2.imread(image_path)
                raw_image = cv2.imread(image_path)[:,:,::-1]
                image = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])(raw_image[...,::-1].copy())
                image = image.unsqueeze(0).to(device)
                
                prediction(model, i, gcam, image, raw_image, image_name)
                try:
                    prediction(model, i, gcam, image, raw_image, image_name)
                    #model = None
                    #gcam = None
                    #gc.collect()
                except RuntimeError:
                    #torch.cuda.empty_cache()
                    #torch.cuda.max_memory_allocated(device=None)
                    #torch.cuda.max_memory_cached(device=None)
                    # model = None
                    # gcam = None
                    #gc.collect()
                    print("Runtime error check")
                    #time.sleep(5)
                        # model, gcam = model_load()
                        # prediction(model, gcam,image, raw_image, image_name)
                

        print('check')
        


def prediction(model, foldername, gcam, image, raw_image, image_name):
    
    
    with torch.no_grad():
        pp, cc = torch.topk(nn.Softmax(dim=1)(model(image.cuda())), 1)
        prob = 100*float(pp.item())
        pred = get_class_name(cc.item())
    print('{:.1f}%'.format(100*float(pp.item())))
    print(cc.item())
    print(get_class_name(cc.item()))
    
    
    # prob = 1
    # pred = 'test'

    probs, ids = gcam.forward(image)
    ids_ = ids[0, 0].view(1, 1).to(device)
    gcam.backward(ids = ids_)
    
    
    #target_layers = [models.load_layer(model_type)]

    for target_layer in target_layers:
        print(target_layers)
        regions = gcam.generate(target_layer=target_layer)
        #filename=os.path.join(output_dir, model_name.split('.')[0]+'_'+image_name.split('.')[0]+'_'+str(target_layer)+".png"
               
        output_dir2 = './CAM/' + model_name.split('.')[0] + '/' + foldername
        print('output_dir:', output_dir2)
        save_gradcam(filename=os.path.join(output_dir2, foldername+'_'+get_class_name(cc.item())+'_'+image_name.split('.')[0]+'_'+str(target_layer)+".png"),
                gcam=regions[0, 0],
                raw_image=raw_image, 
                prob=prob,
                prediction=pred,
                paper_cmap = True)
    print(image_name)
    # del image
    # del raw_image
        
    
    #time.sleep(5)


if __name__ == "__main__":
    model, gcam = model_load()
    input_image(model, gcam)
    
    print("Yey Done!!")

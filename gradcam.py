"""This module produces a Grad-CAM image.

This module produces a Grad-CAM image of the input image and the trained model. The model should be
a pytorch model type. The input model makes a result from the input image while collecting gradient
values at the same time. Then, the gradients at the target layer are transformed into color map.
Finally, a Grad-CAM image is produced and saved at the designated path.

The source codes of this module were written by Kazuto Nakashima, and modified by Hyunseok Oh.
The original version can be found in http://kazuto1011.github.io.
The following is the docstring from the original code.

#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26


Several python modules should be pre-installed to utilize this module.
Required modules are as follows:
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- OpenCV


Functions:
    init_gradcam(model)
    load_image(img_path)
    cal_gradcam(model, gcam, image, target_layer)
    save_gradcam(file_path, region, raw_image, paper_cmap(opt))
    single_gradcam(gcam, target_layer, img_path, gcam_path, paper_cmap(opt))

Classes:
    GradCam(model)
"""


import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import matplotlib.cm as cm
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import models

# Original code
class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


# Original code
class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, self.image_shape, mode="bilinear", align_corners=False)
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)
        return gcam


def init_gradcam(model):
    """Initialize a GradCAM instance for the input model

    Args:
        model (PyTorch model): Trained PyTorch model file

    Returns:
        gcam (GradCAM): GradCAM instance for generating Grad CAM images
    """

    model.eval()

    for param in model.parameters():
        param.requires_grad = True

    gcam = GradCAM(model=model)
    return gcam


def load_image(img_path):
    """Load an image and transform into pytorch tensor with normalization

    Args:
        img_path (str): Path of the original image

    Returns:
        image (PyTorch tensor): Pytorch tensor of the transformed image with normalization (ImageNet stat)
        raw_image (ndarray): Numpy array of the raw input image
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #raw_image = cv2.imread(img_path)

    raw_image = Image.open(img_path)
    raw_image = raw_image.resize((224, 224))
    raw_image = raw_image.convert('RGB')
    
    loader = transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = loader(raw_image).unsqueeze(0).to(device)
    return image, raw_image
    #image = transforms.Compose(
    #    [
    #        transforms.ToTensor(),
    #        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #    ]
    #)(raw_image[..., ::-1].copy())
    #image = image.unsqueeze(0).to(device)
    #return image, raw_image

def raw_load_image(img_path):
    """Load an image and transform into pytorch tensor with normalization

    Args:
        img_path (str): Path of the original image

    Returns:
        image (PyTorch tensor): Pytorch tensor of the transformed image with normalization (ImageNet stat)
        raw_image (ndarray): Numpy array of the raw input image
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #raw_image = cv2.imread(img_path)

    #change cv2 to Image library
    raw_image = Image.open(img_path)
    raw_image = raw_image.convert('RGB')
    loader = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = loader(raw_image).unsqueeze(0).to(device)
    return image, raw_image

def cal_gradcam(gcam, image, target_layer):
    """Calculate the gradients and extract information at the target layer

    Args:
        gcam (GradCAM): GradCAM instance for generating Grad CAM images
        image (PyTorch tensor): Pytorch tensor of the transformed image with normalization (ImageNet stat)
        target_layer (str): Name of the target layer of the model (Must have the same layer name in the model)

    Returns:
        region (ndarray): Grad CAM values of the input image at the target layer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs, ids = gcam.forward(image)
    ids_ = ids[0, 0].view(1, 1).to(device)

    gcam.backward(ids=ids_)

    regions = gcam.generate(target_layer=target_layer)
    return regions[0, 0], probs[0,0].item(), ids[0,0].item()

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
    for i, key in enumerate(['blue', 'green', 'red']):
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

def save_gradcam(file_path, region, raw_image, org_raw_image, prob, prediction, dir, labels, paper_cmap=False):
    """Save the Grad CAM image

    Args:
        file_path (str): Path that the Grad CAM image will be saved at
        region (ndarray): Grad CAM values of the input image at the target layer
        raw_image (ndarray): Numpy array of the raw input image
        paper_cmap (bool)(opt): Determine whether the color map is drawn throughout or in part of the image
                                False: Throughout / True: Part

    Returns:
    """
    org_raw_image = np.array(org_raw_image)[:,:,::-1]
    raw_image = np.array(raw_image)[:,:,::-1]
    # add light_jet
    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)

    region = region.cpu().numpy()

    cmap = light_jet(region)[..., :3] *  255.0
    #cmap = cm.jet_r(region)[..., :3] * 255.0
    if paper_cmap:
        alpha = region[..., None]
        region = alpha * cmap + (1 - alpha) * raw_image ## if you need raw images using preprocessing
        #region = alpha * cmap + (1 - alpha) * org_raw_image ## if you need original raw images except for not preprocessing
    else:
        #region = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2 ## if you need raw images using preprocessing
        region = (cmap.astype(np.float) + org_raw_image.astype(np.float)) / 2 ## if you need original raw images except for not preprocessing

    #labels = ['CIN','Non-Neoplasm']
    index = int(prediction)
    print(index)
    labelname = labels[index]

    #labelprediction = dir[prediction]

    labelprediction = dir[index]

    print('pred label: ', labelprediction)

    # print('pppppppppppath', file_path.split('_')[1])

    prob = prob * 100
    prob_str = round(prob,1)
    prob_str = str(prob_str)

    file_path = file_path.split('_')[0] + "_" + labelname + "_" + prob_str + "_" + file_path.split('_',1)[1]
        
    #print('kkkkkkkkkk',file_path)

    #print(dir)


    print('-------- file path: ', file_path)

    #prediction and probability
    # prob = prob * 100
    
    ##################################################### Original code [:,:,::-1] #####################################################
    #cv2.imwrite(file_path, np.uint8(region))

    
    ##################################################### matplot result ################################################################
    plt.imshow(np.uint8(region)[:,:,::-1])#
    plt.title('{}: {:.1f}%'.format(labelname, prob))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_path,bbox_inces='tight',pad_inches=0,dpi=100)


def single_gradcam(gcam, target_layer, img_path, raw_img_path, gcam_path, dir, labels, paper_cmap=True):
    """Make a single Grad CAM image at once. Execute load_image, cal_gradcam, and save_gradcam at once

    Args:
        gcam (GradCAM): GradCAM instance for generating Grad CAM images
        target_layer (str): Name of the target layer of the model (Must have the same layer name in the model)
        img_path (str): Path of the original image
        gcam_path (str): Path that the Grad CAM image will be saved at
        paper_cmap (bool)(opt): Determine whether the color map is drawn throughout or in part of the image
                                False: Throughout / True: Part

    Returns:
    """

    image, raw_image = load_image(img_path)
    _, org_raw_image = raw_load_image(raw_img_path)
    region, probs, pred = cal_gradcam(gcam, image, target_layer)
    #pred = pred +1
    save_gradcam(file_path=gcam_path, region=region, raw_image=raw_image, org_raw_image= org_raw_image, prob=probs, prediction=pred, dir=dir, labels=labels, paper_cmap=paper_cmap)


def main():

    model_type = "DenseNet161" #Resnet101, IRnet, Efficientnet, NASNetALarge, DenseNet161, DenseNet169
    model_path = "/home/huray/workspace/classifier-pytorch/models/food_bin_7/DenseNet_best_state_for_300epoch.pth"
    #target_layer = 'conv2d_7b'
    labels = ['food','nonfood']

    model = models.load(model_type, num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    model.eval()

    print(model)

    ########## Case 1: Single file ##########
    # img_path = "./test.jpg"
    # result_path = "./test_gcam.jpg"

    # gcam = init_gradcam(model)
    # single_gradcam(gcam, target_layer, img_path, result_path, paper_cmap=True)

    ########## Case 2: Multiple files in a directory ##########

    import os
    #IRnet: conv2d_7b, DenseNet161: features, EfficientNet: _bn1
    target_layers = ["features"]#, "layer2", "layer3", "layer4"]

    img_folder = "/home/huray/workspace/data/number_7/"
    result_folder = "./CAM/"

    #labels = ['CIN','Non-Neoplasm']

    gcam = init_gradcam(model)
    print('check')

    for (path, dir, files) in os.walk(img_folder):
        print(path)
        dir.sort()
        print('dir',dir)
        for i in dir:
            img_labelfolder = img_folder + i
            print(img_labelfolder)
            for idx, img in enumerate(os.listdir(img_labelfolder)):
            #print('check: ',dir)
                img_path = os.path.join(img_labelfolder, img)
                #raw_img_labelfolder = img_labelfolder + '/raw/'
                raw_img_labelfolder = '/home/huray/workspace/data/ext_number_7/0'
                raw_img_path = os.path.join(raw_img_labelfolder, img)
                print('image_path:', img_path)
                print('raw_image_path:', raw_img_path)
                if os.path.isdir(img_path):
                    continue
                print(img)
                result_labelfolder = result_folder + i
                print('res',result_labelfolder)

                try:
                    if not os.path.exists(result_labelfolder):
                        os.makedirs(result_labelfolder)
                except OSError:
                    print("Error: Creating directory. " )
                index = int(i)
                labelname = labels[index]

                for tidx, target_layer in enumerate(target_layers):
                    #print(target_layer)
                    # result_path = os.path.join(
                    #     result_labelfolder, img.split(".")[0] + "_" + i + "_" + target_layer + ".jpg"
                    # )
                    result_path = os.path.join(
                        result_labelfolder, labelname + "_" + img.split(".")[0] + ".jpg"
                    )
                    print(result_path)
                    single_gradcam(gcam, target_layer, img_path, raw_img_path, result_path, dir, labels, paper_cmap=True)

                print("{} / {} Finished".format(idx, len(os.listdir(img_folder))))


if __name__ == "__main__":
    main()

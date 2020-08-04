import torch
import torchvision.models as tm
import torch.nn as nn
import pretrainedmodels as pm
from efficientnet_pytorch import EfficientNet

def resnet101(pretrained = False, fine_tune = False, num_classes = 9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        model = tm.resnet101(pretrained = pretrained)

        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

    else:
        model = tm.resnet101(pretrained = pretrained, num_classes = num_classes).to(device)

    return model


def irnet(pretrained = False, fine_tune = False, num_classes = 9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        model = pm.inceptionresnetv2(num_classes = 1000, pretrained = 'imagenet')

        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.last_linear.in_features
        model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        model.last_linear = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
    else:
        model = pm.inceptionresnetv2(num_classes = num_classes, pretrained = None).to(device)

    return model    

def efficientnet(pretrained = False, fine_tune = False, num_classes = 9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b5')
        if not fine_tune:
                for param in model.parameters():
                    param.requires_grad = False
        
    else:
        model = EfficientNet.from_name('efficientnet-b5')

    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model

def NASNetALarge(pretrained = False, fine_tune = False, num_classes = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        model = pm.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')
        if not fine_tune:
                for param in model.parameters():
                    param.requires_grad = False
        num_ftrs = model.last_linear.in_features
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
    else:
        model = pm.__dict__['nasnetalarge'](num_classes= num_classes, pretrained= None)
    
    return model

def DenseNet161(pretrained = False, fine_tune = False, num_classes = 6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        model = tm.densenet161(pretrained = pretrained)

        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

    else:
        model = tm.densenet161(pretrained = pretrained, num_classes = num_classes).to(device)

    return model

def DenseNet169(pretrained = False, fine_tune = False, num_classes = 6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained:
        model = tm.densenet169(pretrained = pretrained)

        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

    else:
        model = tm.densenet169(pretrained = pretrained, num_classes = num_classes).to(device)

    return model

def load(name, pretrained = True, num_classes = 6):
    if name == 'Resnet101':
        return resnet101(pretrained, True, num_classes)
    elif name == 'IRnet':
        return irnet(pretrained, True, num_classes)
    elif name == 'Efficientnet':
        return efficientnet(pretrained, True, num_classes)
    elif name == 'NASNetALarge':
        return NASNetALarge(pretrained, True, num_classes)
    elif name == 'DenseNet161':
        return DenseNet161(pretrained, True, num_classes)
    elif name == 'DenseNet169':
        return DenseNet169(pretrained, True, num_classes)
    else:
        raise Exception('Invalid model name')

def load_layer(name):
    #print(name)
    layer_dict = {"Resnet101" : 'layer4', "IRnet" : 'conv2d_7b', "EfficientNet" : '_bn1', "NASNetALarge" : 'cell_17.conv_1x1.conv', 
                 "DenseNet161": 'features', "DenseNet169": 'features'}

    if name in layer_dict:
        return layer_dict[name]
    else:
        raise Exception('Invalid model name')

# model = resnet101(True, True, 9)
# print(model)
# f = open("IRnet.info","w")
# f.write(str(model))

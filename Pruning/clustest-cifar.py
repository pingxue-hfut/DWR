# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:13:38 2020

@author: ASUS
"""

import torch
from torch.autograd import Variable
import torch.nn as nn


import os
import random
from sklearn.cluster import DBSCAN

import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import resnet18
from modules import ir_1w1a
# import ir_1w1a
import argparse
import vgg
import resnet
import vgg11



parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
global args
args = parser.parse_args()

loss_func = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg16': 13,
    'resnet56' : 27,
    'resnet110' : 54,
    'googlenet' : 9,
    'densenet':36,
    }

channels = 4

# origin_model = vgg.__dict__['vgg_small_1w1a']()
# origin_model = resnet.__dict__['resnet20_1w1a']()
# origin_model = resnet18.__dict__['ResNet18']()
origin_model = vgg11.__dict__['vgg_11']()
checkpoint = torch.load('pretrained/ReActNet/vgg11/model.th')
origin_model = nn.DataParallel(origin_model).cuda()
origin_model.load_state_dict(checkpoint['state_dict'], strict=False)
for p in origin_model.parameters():
    p.requires_grad = False
origin_model.eval()

fmap_block = []

def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)


# register hook
i = 0
for m in origin_model.modules():
    if isinstance(m, ir_1w1a.IRConv2d):
        i = i + 1
        '''
        if i % 2 == 0:
            m.register_forward_hook(forward_hook)
        '''
        if i != 5:
            m.register_forward_hook(forward_hook)



transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
testset = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test))

image = []
for images, labels in testset:
    # images = Variable(torch.unsqueeze(images, dim=0).float(), requires_grad=False)
    images = Variable(images.float(), requires_grad=False)
    image.append(images)
#image = image.cuda()
print(np.array(image[3]).shape)

for i in random.sample(range(10000), 400):
    imagetest = image[i].cuda()
    with torch.no_grad():
        outputs = origin_model(imagetest)


# get feature_map with size of (batchsize, channels, W, H)
feature_map = []
for k in range(channels):
    feature_map.append(fmap_block[k])


for c in range(channels):
    for j in np.arange(c+channels, len(fmap_block), channels):
        feature_map[c] = torch.cat((feature_map[c], fmap_block[j]), dim=0)

netchannels = torch.zeros(channels)
for s in range(channels):
    # print(feature_map[s].shape)
    # change the size of feature_map from (batchsize, channels, W, H) to (batchsize, channels, W*H)
    a, b, c, d = feature_map[s].size()
    feature_map[s] = feature_map[s].view(a, b, -1)
    #print(feature_map[s].shape)
    # 
    feature_map[s] = torch.sum(feature_map[s], dim=0)/a
    #print(feature_map[s].shape)


    
    # clustering
    X = np.array(feature_map[s].cpu())
    clustering = DBSCAN(eps=0.021, min_samples=5, metric='cosine').fit(X)

    # defult: eps=0.5, min_samples=5
    # ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
    labels = clustering.labels_

    #print(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    netchannels[s] = netchannels[s]+n_clusters_+n_noise_

    #print('Estimated number of clusters: %d' % n_clusters_)
    #print('Estimated number of noise points: %d' % n_noise_)

netchannels = np.array(netchannels)

print(netchannels)
    
#print(feature_map[2])



#print(fmap_block[0])







'''
def test(model, testLoader):
    global best_acc, glb_feature
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # register hook
            for m in origin_model.modules():
                if isinstance(m, nn.Conv2d):
                    m.register_forward_hook(forward_hook)
                    #handle.remove()
            loss = loss_func(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        
        print(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg

test(origin_model, loader.testLoader)
'''



#print(origin_model.features)























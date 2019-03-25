use_gpu = True
#This is a new line

import time
import os
import copy
import argparse
import pdb
import collections
import sys
import pickle
import ntpath
import cv2
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CustomDataset,CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def train(img_dir,classes_csv,model_fname=None,resnet_depth=50,epochs=1000,steps=100,train_split=0.8,out_dir ='',out_prefix=''):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the data loaders

    # Get all image fnames in folder
    img_list = []
    if not isinstance(img_dir, list):
        img_dir = [img_dir]
    for dir in img_dir:
        for file in os.listdir(dir):
            if file.endswith(".png"):
                img_list.append(dir + file)

    randomised_list = random.sample(img_list, len(img_list))
    num_train = int(0.8*len(img_list))
    train_imgs, val_imgs = randomised_list[:num_train], randomised_list[num_train:]

    dataset_train = CustomDataset(img_list=train_imgs, class_list=classes_csv, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CustomDataset(img_list=val_imgs, class_list=classes_csv,transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model

    if resnet_depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif resnet_depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # retinanet = torch.load(model_fname)

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    if model_fname is not None:
        retinanet.load_state_dict(torch.load(model_fname))

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()

    start_time = time.clock()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                # print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        print('Epoch: {} | Running loss: {:1.5f} | Elapsed Time: {}'.format(epoch_num, np.mean(loss_hist),(time.clock() - start_time)/60))
        mAP = csv_eval.evaluate(dataset_val, retinanet)
        scheduler.step(np.mean(epoch_loss))

        if (epoch_num) % steps == 0:
            torch.save(retinanet.module, '{}{}_model_{}.pt'.format(out_dir, out_prefix, epoch_num))
            torch.save(retinanet.state_dict(), '{}{}_state_{}.pt'.format(out_dir, out_prefix, epoch_num))

    torch.save(retinanet, out_dir + '{}model_final.pt'.format(out_prefix))
    torch.save(retinanet.state_dict(), out_dir + '{}state_final_.pt'.format(out_prefix))


def infer(img_dir,classes_csv,model_fname,resnet_depth,score_thresh,out_dir, results_fname):

    # Create dataset
    img_list = []
    if not isinstance(img_dir, list):
        img_dir = [img_dir]
    for dir in img_dir:
        for file in os.listdir(dir):
            if file.endswith(".png"):
                img_list.append(dir + file)

    dataset_val = CustomDataset(img_list=img_list, class_list=classes_csv, transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    print(dataset_val.num_classes())

    # Create the model
    if resnet_depth == 18:
        retinanet = model.resnet18(num_classes=dataset_val.num_classes())
    elif resnet_depth == 34:
        retinanet = model.resnet34(num_classes=dataset_val.num_classes())
    elif resnet_depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes())
    elif resnet_depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes())
    elif resnet_depth == 152:
        retinanet = model.resnet152(num_classes=dataset_val.num_classes())
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    state_dict = torch.load(model_fname)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    retinanet.load_state_dict(new_state_dict)

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()
    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    results = []

    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('Elapsed time: {}, Num objects: {}'.format(time.time() - st, len(scores)))

            idxs = np.where(scores > score_thresh)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0)).astype(np.uint8).copy()

            bboxes = []
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0] / data['scale'][0])
                y1 = int(bbox[1] / data['scale'][0])
                x2 = int(bbox[2] / data['scale'][0])
                y2 = int(bbox[3] / data['scale'][0])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                score = float(scores[idxs[0][j]])

                bboxes.append([x1, y1, x2, y2, score])

            img_fname = ntpath.basename(data['img_fname'][0])
            results.append([img_fname, bboxes])
    #         fig, ax = plt.subplots(figsize=(12, 12))
    #         ax.imshow(img, interpolation='bilinear')

    with open(out_dir+results_fname,"wb") as output_file:
        pickle.dump(results, output_file)


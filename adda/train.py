#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import copy

from image_dataset import ImageDataset
from models import LeNetEncoder, LeNetClassifier, Discriminator, init_weights


# In[2]:


########## Arguments ##########
num_epochs =10
src_num_epochs = 30
tgt_num_epochs = 100
src_lr = 5e-4
tgt_lr = 2e-4
lr=1e-4
batch_size=128

# svhn, usps, mnistm

d_source = "mnistm"
d_target = "svhn"

output_src_encoder_path = "./models/src_encoder_{}_{}.pth".format(d_source, d_target)
output_src_classifier_path = "./models/src_classifier_{}_{}.pth".format(d_source, d_target)
output_tgt_encoder_path = "./models/tgt_encoder_{}_{}.pth".format(d_source, d_target)
output_discriminator_path = "./models/discirminator_{}_{}.pth".format(d_source, d_target)


print("### [Info] Source: {} | Target: {} ###".format(d_source, d_target))
#############################


# In[3]:


# load datasets
src_dataset = ImageDataset("train", d_source)
tgt_dataset = ImageDataset("train", d_target)
val_src_dataset = ImageDataset("val", d_source)
val_tgt_dataset = ImageDataset("val", d_target)

src_dataloader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)
tgt_dataloader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True)
val_src_dataloader = DataLoader(val_src_dataset, batch_size=batch_size, shuffle=False)
val_tgt_dataloader = DataLoader(val_tgt_dataset, batch_size=batch_size, shuffle=False)

src_label = 0
tgt_label = 1

print(len(src_dataset), len(tgt_dataset), len(val_src_dataset), len(val_tgt_dataset))


# In[4]:


# init models
src_encoder = LeNetEncoder()
tgt_encoder = LeNetEncoder()
src_classifier = LeNetClassifier()
discriminator = Discriminator()

# init weights
src_encoder.apply(init_weights)
tgt_encoder.apply(init_weights)
src_classifier.apply(init_weights)
discriminator.apply(init_weights)

# to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_encoder.to(device)
tgt_encoder.to(device)
src_classifier.to(device)
discriminator.to(device)


# In[5]:


# print("### Source Encoder ###")
# print(src_encoder)
# print("### Source Classifier ###")
# print(src_classifier)
# print("### Targer Encoder ###")
# print(tgt_encoder)
# print("### discriminator ###")
# print(discriminator)


# In[6]:


optimizer = optim.Adam(
    list(src_encoder.parameters()) + list(src_classifier.parameters()),
    lr=src_lr,
    betas=(0.5, 0.9))
optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
    lr=lr,
    betas=(0.5, 0.9))
optimizer_discriminator = optim.Adam(discriminator.parameters(),
    lr=lr,
    betas=(0.5, 0.9))

criterion = nn.CrossEntropyLoss()

_len = min(len(src_dataloader), len(tgt_dataloader))


# In[7]:


def train_src_one_epoch(trange):
    running_loss = 0.0
    running_corrects = 0.0
    
    for images, labels in trange:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # compute loss for critic
        preds = src_classifier(src_encoder(images))
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        # calculate accuracy
        _, pred_labels = torch.max(preds, 1)
        corrects = torch.sum(pred_labels == labels.data)

        # log
        running_loss += loss.item() * labels.shape[0]
        running_corrects += corrects
        
        postfix_dict = {
            "loss": "{:.5f}".format(loss.item()),
            "acc": "{:.5f}".format(corrects.double() / labels.shape[0]),
        }
        trange.set_postfix(**postfix_dict)
    
    return running_loss, running_corrects
    


# In[8]:


def eval_src_one_epoch(eval_trange):
    eval_loss = 0.0
    eval_corrects = 0.0
    
    for images, labels in eval_trange:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            preds = src_classifier(src_encoder(images))
            loss = criterion(preds, labels)
        
        # calculate accuracy
        _, pred_labels = torch.max(preds, 1)
        corrects = torch.sum(pred_labels == labels.data)
        eval_loss += loss.item() * labels.shape[0]
        eval_corrects += corrects
        
        # log
        postfix_dict = {
            "loss": "{:.5f}".format(loss.item()),
            "acc": "{:.5f}".format(corrects.double() / labels.shape[0]),
        }

        eval_trange.set_postfix(**postfix_dict)
   
    return eval_loss, eval_corrects


# In[9]:


def train_tgt_one_epoch(trange):
    running_loss_tgt = 0.0
    running_loss_discriminator = 0.0
    
    for (sources, targets) in trange:
        source_images, labels = sources[0].to(device), sources[1].to(device)
        target_images = targets[0].to(device)
        
        source_bs = source_images.shape[0]
        target_bs = target_images.shape[0]

        ### train discriminator
        optimizer_discriminator.zero_grad()
        
        source_feature = src_encoder(source_images)
        target_feature = tgt_encoder(target_images)
        feature_concat = torch.cat((source_feature, target_feature), 0)
                
        # predict on discriminator
        pred_concat = discriminator(feature_concat.detach())
    
        # prepare real and fake label
        source_labels = torch.full((source_bs,), src_label, device=device).long()
        target_labels = torch.full((target_bs,), tgt_label, device=device).long()
        label_concat = torch.cat((source_labels, target_labels), 0)

        # compute loss for critic
        loss_discriminator = criterion(pred_concat, label_concat)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        # calculate accuracy
        _, pred_labels = torch.max(pred_concat, 1)
        corrects = torch.sum(pred_labels == label_concat.data)
        
        ### train target encoder
        optimizer_discriminator.zero_grad()
        optimizer_tgt.zero_grad()
        
        target_feature = tgt_encoder(target_images)
        target_pred = discriminator(target_feature)
        
        # train with fake (source) label
        fake_target_labels = torch.full((target_bs,), src_label, device=device).long()
        loss_tgt = criterion(target_pred, fake_target_labels)
        loss_tgt.backward()
        optimizer_tgt.step()
        
        running_loss_tgt += loss_tgt.item()
        running_loss_discriminator += loss_discriminator.item()

        # log
        postfix_dict = {
            "loss discriminator": "{:.5f}".format(loss_discriminator.item()),
            "domain acc": "{:.5f}".format(corrects.double() / len(label_concat)),
            "loss tgt": "{:.5f}".format(loss_tgt.item())
        }
        trange.set_postfix(**postfix_dict)
        
    return running_loss_tgt, running_loss_discriminator


# In[10]:


def eval_tgt_one_epoch(eval_trange):
    eval_loss = 0.0
    eval_corrects = 0.0
    
    for images, labels in eval_trange:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            preds = src_classifier(tgt_encoder(images))
            loss = criterion(preds, labels)
        
        # calculate accuracy
        _, pred_labels = torch.max(preds, 1)
        corrects = torch.sum(pred_labels == labels.data)
        eval_loss += loss.item() * labels.shape[0]
        eval_corrects += corrects
        
        # log
        postfix_dict = {
            "loss": "{:.5f}".format(loss.item()),
            "acc": "{:.5f}".format(corrects.double() / labels.shape[0]),
        }

        eval_trange.set_postfix(**postfix_dict)

    return eval_loss, eval_corrects


# In[11]:


def eval_src_only_one_epoch(eval_trange):
    eval_loss = 0.0
    eval_corrects = 0.0
    
    for images, labels in eval_trange:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            preds = src_classifier(src_encoder(images))
            loss = criterion(preds, labels)
        
        # calculate accuracy
        _, pred_labels = torch.max(preds, 1)
        corrects = torch.sum(pred_labels == labels.data)
        eval_loss += loss.item() * labels.shape[0]
        eval_corrects += corrects
        
        # log
        postfix_dict = {
            "loss": "{:.5f}".format(loss.item()),
            "acc": "{:.5f}".format(corrects.double() / labels.shape[0]),
        }

        eval_trange.set_postfix(**postfix_dict)

    return eval_loss, eval_corrects


# In[12]:


best_acc = 0.0
best_src_encoder = None
best_src_classifier = None

best_tgt_acc = 0.0
best_tgt_encoder = None
best_discriminator = None

acc_arr = []

print("### [Info] Training encoder and classifier for source ###")

for epoch in range(src_num_epochs):
    print("Epoch {}/{}".format(epoch + 1, src_num_epochs))

    # train model on source
    src_encoder.train()
    src_classifier.train()
    trange = tqdm(src_dataloader)
    running_loss, running_corrects = train_src_one_epoch(trange)
    
    # eval model on source
    src_encoder.eval()
    src_classifier.eval()        
    eval_trange = tqdm(val_src_dataloader)
    eval_loss, eval_corrects = eval_src_one_epoch(eval_trange)
    
    epoch_loss = running_loss / len(src_dataset)
    epoch_acc = running_corrects / len(src_dataset)
            
    eval_epoch_loss = eval_loss / len(val_src_dataset)
    eval_epoch_acc = eval_corrects / len(val_src_dataset)
    
    print("Train | Loss: {:.5f} | Accuracy: {:.5f}".format(epoch_loss, epoch_acc))
    print("Val | Loss: {:.5f} | Accuracy: {:.5f}".format(eval_epoch_loss, eval_epoch_acc))
    
    if eval_epoch_acc > best_acc:
        best_acc = eval_epoch_acc
        best_src_encoder = copy.deepcopy(src_encoder.state_dict())
        best_src_classifier = copy.deepcopy(src_classifier.state_dict())
        

# eval model on target
eval_trange = tqdm(val_tgt_dataloader)
eval_tgt_loss, eval_tgt_corrects = eval_src_only_one_epoch(eval_trange)
            
eval_epoch_tgt_loss = eval_tgt_loss / len(val_tgt_dataset)
eval_epoch_tgt_acc = eval_tgt_corrects / len(val_tgt_dataset)
print("Val | Tgt Loss: {:.5f} | Tgt Accuracy: {:.5f}".format(eval_epoch_tgt_loss, eval_epoch_tgt_acc))
acc_arr.append(eval_epoch_tgt_acc)

best_tgt_acc = eval_epoch_tgt_acc
best_tgt_encoder = best_src_encoder
best_discriminator = None
    
# init target encoder with source encoder
if best_src_encoder is not None:
    tgt_encoder.load_state_dict(best_src_encoder)
    
print("### [Info] Training target encoder by GAN ###")
for epoch in range(tgt_num_epochs):
    print("Epoch {}/{}".format(epoch + 1, tgt_num_epochs))
    
    # train model on target
    tgt_encoder.train()
    discriminator.train()    
    trange = tqdm(zip(src_dataloader, tgt_dataloader), total=_len)
    running_loss_tgt, running_loss_discriminator = train_tgt_one_epoch(trange)

    # eval model on target
    tgt_encoder.eval()
    discriminator.eval()
    eval_trange = tqdm(val_tgt_dataloader)
    eval_loss, eval_corrects = eval_tgt_one_epoch(eval_trange)
    
    eval_epoch_loss = eval_loss / len(val_tgt_dataset)
    eval_epoch_acc = eval_corrects / len(val_tgt_dataset)

    print("Train | Target Loss: {:.5f} |  Discriminator Loss: {:.5f}".format(running_loss_tgt/_len, running_loss_discriminator/_len))
    print("Val | Target Loss: {:.5f} | Target Accuracy: {:.5f}".format(eval_epoch_loss, eval_epoch_acc))
    acc_arr.append(eval_epoch_acc)
    if eval_epoch_acc > best_tgt_acc:
        best_tgt_acc = eval_epoch_acc
        best_tgt_encoder = copy.deepcopy(tgt_encoder.state_dict())
        best_discriminator = copy.deepcopy(discriminator .state_dict())
        
# save model        
if best_src_encoder is not None:
    torch.save(best_src_encoder, output_src_encoder_path)
    print("Src Encoder saved")
if best_src_classifier is not None:
    torch.save(best_src_classifier, output_src_classifier_path)
    print("Src Classifier saved")        
if best_tgt_encoder is not None:
    torch.save(best_tgt_encoder, output_tgt_encoder_path)
    print("Tgt Encoder saved")
if best_discriminator is not None:
    torch.save(best_discriminator, output_discriminator_path)
    print("Discriminator saved")
    
    


# In[ ]:


np.save('target_acc_{}_{}.npy'.format(d_source, d_target), np.array(acc_arr))


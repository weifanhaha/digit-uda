#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
import copy

from models import DANN
from image_dataset import ImageDataset


# In[2]:


########## Arguments ##########
num_epochs = 15
lr = 1e-3
batch_size = 128

# svhn, usps, mnistm
d_source = "svhn"
d_target = "usps"

train_with_domain = True
train_with_target = False

assert (train_with_domain and train_with_target) == False

if train_with_domain:
    output_model_path = "./models/{}_{}_domain.pth".format(d_source, d_target)
elif train_with_target:
    output_model_path = "./models/{}_{}_target.pth".format(d_source, d_target)
else:
    output_model_path = "./models/{}_{}.pth".format(d_source, d_target)

#############################


# In[3]:


source_dataset = ImageDataset("train", d_source)
target_dataset = ImageDataset("train", d_target, is_target=True)
val_dataset = ImageDataset("val", d_target, is_target=True)

source_dataloader = DataLoader(
    source_dataset, batch_size=batch_size, shuffle=True)
target_dataloader = DataLoader(
    target_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

source_label = 0
target_label = 1


# In[4]:


print(len(source_dataset), len(target_dataset), len(val_dataset))


# In[5]:


model = DANN(drop_p=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[6]:


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)

# Criterion
class_criterion = nn.NLLLoss()
domain_criterion = nn.NLLLoss()


# In[7]:


label_losses = []
domain_losses = []


# In[8]:


def train_one_epoch(enum_dataloader, start_steps, total_steps, train_with_target=False):

    running_loss = 0.0
    running_corrects = 0.0

    for idx, (sources, targets) in enum_dataloader:
        source_images, labels = sources[0].to(device), sources[1].to(device)
        target_images = targets[0].to(device)

        source_bs = source_images.shape[0]
        target_bs = target_images.shape[0]

        source_labels = torch.full(
            (source_bs,), source_label, device=device).long()
        target_labels = torch.full(
            (target_bs,), target_label, device=device).long()

        p = (idx + start_steps) / total_steps
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        label_output, domain_output = model(source_images, alpha)
        label_loss = class_criterion(label_output, labels)
        running_loss += label_loss.item() * source_bs

        # calculate label acc
        _, pred_labels = torch.max(label_output, 1)
        corrects = torch.sum(pred_labels == labels.data)
        running_corrects += corrects

        # calculate domain loss
        if train_with_domain:
            _, domain_t_output = model(target_images, alpha)
            domain_s_loss = domain_criterion(domain_output, source_labels)
            domain_t_loss = domain_criterion(domain_t_output, target_labels)
            domain_loss = domain_s_loss + domain_t_loss
            running_loss += domain_loss.item() * source_bs
        else:
            domain_loss = 0.0

        # update parameters
        loss = label_loss + domain_loss
        loss.backward()
        optimizer.step()

        # record
        label_losses.append(label_loss)
        domain_losses.append(domain_loss)

        postfix_dict = {
            "L/D": "{:.5f}/{:.5f}".format(label_loss, domain_loss),
            "SD / TD": "{:5f}/{:5f}".format(domain_s_loss, domain_t_loss),
            "acc": "{:.5f}".format(corrects.double() / source_bs),
            "alpha": "{:.5f}".format(alpha),
        }

        enum_dataloader.set_postfix(**postfix_dict)

    return running_loss, running_corrects


# In[9]:


def eval_one_epoch(model, enum_dataloader):
    running_loss = 0.0
    running_corrects = 0.0

    for idx, targets in enum_dataloader:
        target_images, labels = targets[0].to(device), targets[1].to(device)
        target_bs = target_images.shape[0]

        optimizer.zero_grad()
        with torch.no_grad():
            label_output, domain_output = model(target_images, 1.0)
            label_loss = class_criterion(label_output, labels)
            loss = label_loss
            running_loss += label_loss.item() * target_bs

        # calculate label acc
        _, pred_labels = torch.max(label_output, 1)
        corrects = torch.sum(pred_labels == labels.data)
        running_corrects += corrects

        postfix_dict = {
            "L": "{:.5f}".format(label_loss),
            "acc": "{:.5f}".format(corrects.double() / target_bs)
        }

        enum_dataloader.set_postfix(**postfix_dict)

    return running_loss, running_corrects


# In[10]:


best_acc = 0.0
best_model = None

losses = []

for epoch in range(num_epochs):
    model.train()

    if not train_with_target:
        _len = min(len(source_dataloader), len(target_dataloader))
        trange = tqdm(
            enumerate(zip(source_dataloader, target_dataloader)),
            total=_len,
            desc="Epoch {}".format(epoch),
        )
    else:
        _len = len(target_dataloader)
        # we don't need the second dalaloader in enumerate
        trange = tqdm(
            enumerate(zip(target_dataloader, target_dataloader)),
            total=_len,
            desc="Epoch {}".format(epoch),
        )

    start_steps = epoch * _len
    total_steps = num_epochs * _len

    running_loss, running_corrects = train_one_epoch(
        trange, start_steps, total_steps)

    epoch_loss = running_loss / min(len(source_dataset), len(target_dataset))
    epoch_acc = running_corrects.double() / min(len(source_dataset), len(target_dataset))

    losses.append(epoch_loss)

    model.eval()

    _val_len = len(val_dataloader)
    trange = tqdm(enumerate(val_dataloader),
                  total=_val_len,
                  desc="Testing Epoch {}".format(epoch),
                  )

    eval_loss, eval_corrects = eval_one_epoch(model, trange)

    eval_epoch_loss = eval_loss / len(val_dataset)
    eval_epoch_acc = eval_corrects.double() / len(val_dataset)

    print("Epoch Loss: {:.5f} |  Accuracy: {:.5f}".format(
        epoch_loss, epoch_acc))
    print("Testing | Label Loss: {:.5f} |  Accuracy: {:.5f}".format(
        eval_epoch_loss, eval_epoch_acc))

    if eval_epoch_acc > best_acc:
        best_acc = eval_epoch_acc
        best_model = copy.deepcopy(model.state_dict())

if best_model != None:
    torch.save(best_model, output_model_path)


# In[ ]:


# from matplotlib import pyplot as plt
# plt.plot(domain_losses)


# In[ ]:

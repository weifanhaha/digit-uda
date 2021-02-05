#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from torch import optim
from torch.utils.data import DataLoader

from models import LeNetEncoder
from image_dataset import ImageDataset


# In[2]:


########## Arguments ##########
batch_size=128

# svhn, usps, mnistm
d_source = "svhn"
d_target = "usps"

output_src_encoder_path = "./models/src_encoder_{}_{}.pth".format(d_source, d_target)
output_tgt_encoder_path = "./models/tgt_encoder_{}_{}.pth".format(d_source, d_target)

    
output_tsne_a_path = "./images/{}_{}_label.png".format(d_source, d_target)    
output_tsne_b_path = "./images/{}_{}_domain.png".format(d_source, d_target)    
#############################


# In[3]:


# prepare dataset
batch_size = 128
source_dataset = ImageDataset("test", d_source)
target_dataset = ImageDataset("test", d_target)

source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare model
# init models
src_encoder = LeNetEncoder()
tgt_encoder = LeNetEncoder()

# to device
src_encoder.to(device)
tgt_encoder.to(device)


# init weights
src_encoder.load_state_dict(torch.load(
        output_src_encoder_path, map_location=device))

tgt_encoder.load_state_dict(torch.load(
        output_tgt_encoder_path, map_location=device))


# In[5]:


# get latent space
latents = []
labels = []
domains = []

src_encoder.eval()
tgt_encoder.eval()

for sources in tqdm(source_dataloader):
    source_images, label = sources[0].to(device), sources[1]
    with torch.no_grad():
        features = src_encoder(source_images)
        latents.append(features.cpu())
    labels.append(label)
    
    
for targets in tqdm(target_dataloader):
    target_images, label = targets[0].to(device), targets[1]
    with torch.no_grad():
        features = tgt_encoder(target_images)
        latents.append(features.cpu())
    labels.append(label)

all_latents = torch.cat(latents)        
all_labels = torch.cat(labels)


# In[6]:


print("Calculating tsne...")
tsne = TSNE(n_components=2)
embeddings = tsne.fit_transform(all_latents)


# In[9]:


print("Ploting tsne...")
source_domains = torch.zeros(len(source_dataset))
target_domains =  torch.ones(len(target_dataset))
domains = torch.cat((source_domains, target_domains))


# In[12]:


c = plt.scatter(embeddings[:,0], embeddings[:,1], c=domains, cmap='RdBu_r', s=6)
plt.legend(*c.legend_elements(), loc='upper right')
plt.savefig(output_tsne_b_path)


# In[13]:


c = plt.scatter(embeddings[:,0], embeddings[:,1], c=all_labels, cmap='RdBu_r', s=6)
plt.legend(*c.legend_elements(), loc='upper right')
plt.savefig(output_tsne_a_path)


# In[15]:





# In[ ]:





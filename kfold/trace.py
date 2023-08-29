# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import gc

import wandb

import functools

from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random
import pickle

from datasets import NACCCurrentDataset
from model import NACCModel
dataset = NACCCurrentDataset("../investigator_nacc57.csv", "../features/combined", fold=0)

dataset[124]
tqdm.pandas()

MODEL = "./models/efficient-dragon-18"
samples = iter(DataLoader(dataset, batch_size=256, shuffle=True))
model = NACCModel(dataset._num_features, 3, nlayers=3, hidden=128)
model.load_state_dict(torch.load(os.path.join(MODEL, "model.save"), map_location=torch.device('cpu')))
model.eval()

X = torch.empty((0, 4)) # [log_mean, log_max, log_median, log_std]
y = torch.empty((0, 2)) # [Wrong, Correct]

for sample in tqdm(samples):
    sample = next(samples)

    preds = model(sample[0], sample[1], sample[2])
    preds = torch.argmax(preds["logits"], dim=1)
    targets = torch.argmax(sample[2], dim=1)
    correct = preds == targets

    intermediate = model.linear0(torch.unsqueeze(sample[0], dim=2))
    attn_out, attn_weight = model.encoder.layers[0].self_attn(intermediate, intermediate, intermediate)
    cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()

    log_attn = torch.log(cross_attn)
    means = torch.mean(torch.mean(log_attn, dim=2), dim=1)
    maxes = torch.max(torch.max(log_attn, dim=2).values, dim=1).values
    medians = torch.median(torch.median(log_attn, dim=2).values, dim=1).values
    stds = torch.std(torch.std(log_attn, dim=2), dim=1)

    results = torch.stack([means, maxes, medians, stds], dim=1)

    X = torch.cat([X, results], dim=0)
    y = torch.cat([y, F.one_hot(correct.to(torch.int64))], dim=0)

y = torch.argmax(y, dim=1)
num_false = sum(y==0)

zero_indx = (y == 0).nonzero(as_tuple=True)[0]
one_indx = (y == 1).nonzero(as_tuple=True)[0]

selections = torch.tensor(random.sample(zero_indx.tolist(), num_false) + 
                          random.sample(one_indx.tolist(), num_false))

X = X[selections]
y = y[selections]

val_samples = torch.tensor([random.random() < 0.1 for _ in range(len(X))])

X_val = X[val_samples]
y_val = y[val_samples]

X = X[~val_samples]
y = y[~val_samples]

data = {
    "train": (X, y),
    "test": (X_val, y_val)
}

with open("../attention_analysis_data_bal.bin", 'wb') as df:
    torch.save(data, df)

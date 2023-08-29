# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd

import gc

import wandb

import functools

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

# import tqdm auto
from tqdm.auto import tqdm
tqdm.pandas()

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random

# model
from tracemodel import TraceModel

CONFIG = {
    "batch_size": 32,
    "lr": 0.0001,
    "epochs": 128,
    "hidden": 256,
}

ONLINE = False
ONLINE = True

run = wandb.init(project="nacc-accuracy", entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))

config = run.config

BATCH_SIZE = config.batch_size
LR = config.lr
EPOCHS = config.epochs
HIDDEN = config.hidden

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# create dataset
with open("../../attention_analysis_data_bal.bin", 'rb') as df:
    data = torch.load(df)

train_data = TensorDataset(data["train"][0], data["train"][1])
test_data = TensorDataset(data["test"][0], data["test"][1])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = iter(DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True))

val_batch = next(test_loader)

# create the model
model = TraceModel(HIDDEN).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# train
for e in range(EPOCHS):
    print(f"Training {e+1}/{EPOCHS}...")

    for i, batch in enumerate(tqdm(iter(train_loader), total=len(train_loader))):
        inp = [i.to(DEVICE) for i in batch]
        output = model(*inp)

        loss = output["loss"]
        preds = output["predictions"]

        acc = sum(preds.int() == inp[1])/len(preds)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        run.log({
            "loss": loss.cpu().item(),
            "acc": acc.cpu().item()
        }) 

        if i % 16 == 0:
            model.eval()

            inp = [i.to(DEVICE) for i in val_batch]
            output = model(*inp)

            loss = output["loss"]
            preds = output["predictions"]

            acc = sum(preds.int() == inp[1])/len(preds)

            run.log({
                "val_loss": loss.cpu().item(),
                "val_acc": acc.cpu().item()
            }) 

            model.train()

# save
model.eval()
os.mkdir(f"./models/{run.name}")
torch.save(model.state_dict(), f"./models/{run.name}/model.save")
torch.save(optimizer, f"./models/{run.name}/optimizer.save")


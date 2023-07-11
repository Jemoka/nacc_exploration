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
from model import NACCModel
# from model_legacy import NACCModel
from datasets import *


CONFIG = {
    "fold": 0,
    "featureset": "combined",
    # "task": "future",
    "task": "current",
    # "base": "treasured-aardvark-129"
    "base": None
}


TASK = CONFIG["task"]

ONE_SHOT = True
# ONE_SHOT = False
ONLINE = False
# ONLINE = True

if ONE_SHOT:
    run = wandb.init(project="nacc_future" if TASK == "future" else "nacc", entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))
else:
    run = wandb.init(project="nacc-kfold", entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))

config = run.config

BATCH_SIZE = 8
EPOCHS = 64
LR = 0.0002
FOLD = config.fold
FEATURESET = config.featureset
MODEL = config.base

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

if TASK == "current":
    dataset = NACCCurrentDataset("../investigator_nacc57.csv", f"../features/{FEATURESET}", fold=FOLD)
elif TASK == "future":
    dataset = NACCFutureDataset("../investigator_nacc57.csv", f"../features/{FEATURESET}", fold=FOLD)
else:
    raise Exception("Weird task heh.")

validation_set = TensorDataset(*dataset.val())
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if not MODEL:
    model = NACCModel(dataset._num_features, 3).to(DEVICE)
else:
    # load model
    model = torch.load(os.path.join("models", MODEL, "model.save"),
                       map_location=DEVICE).to(DEVICE)


optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=8, gamma=0.5)


# get a random validation batch
def val_batch():
    start = random.randint(0, len(validation_set)//BATCH_SIZE)*BATCH_SIZE
    end = start+BATCH_SIZE
    batch = validation_set[start:end]
    return batch

def val():
    model.eval()
    batch = [i.to(DEVICE) for i in val_batch()]
    output = model(batch[0].float(), batch[1], batch[2])
    run.log({"val_loss": output["loss"].detach().cpu().item()})
    model.train()

# calculate the f1 from tensors
def tensor_metrics(logits, labels):
    label_indicies = np.argmax(labels, 1)
    logits_indicies  = logits

    class_names = ["Control", "MCI", "Dementia"]

    pr_curve = wandb.plot.pr_curve(label_indicies, logits_indicies, labels = class_names)
    roc = wandb.plot.roc_curve(label_indicies, logits_indicies, labels = class_names)
    cm = wandb.plot.confusion_matrix(
        y_true=np.array(label_indicies), # can't labels index by scalar tensor
        probs=logits_indicies,
        class_names=class_names
    )

    acc = sum(label_indicies == np.argmax(logits_indicies, axis=1))/len(label_indicies)

    return pr_curve, roc, cm, acc

# model.train()
for epoch in range(EPOCHS):
    print(f"Currently training epoch {epoch}...")

    for i, batch in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
        if i % 64 == 0:
            val()

        batchp = batch
        # send batch to GPU if needed
        batch = [i.to(DEVICE) for i in batch]

        # we skip any batch of 1 element, because of BatchNorm
        if batch[0].shape[0] == 1:
            continue

        # run with actual backprop
        try:
            output = model(batch[0].float(), batch[1], batch[2])
        except RuntimeError:
            optimizer.zero_grad()
            continue

        # backprop
        try:
            output["loss"].backward()
        except RuntimeError:
            breakpoint()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        run.log({"loss": output["loss"].detach().cpu().item()})

    scheduler.step()

# model.eval()

# we track logits and labels and count them
# finally together eventually
logits = np.empty((0,3))
labels = np.empty((0,3))

print("Validating...")

# validation is large, so we do batches
for i in tqdm(iter(validation_loader)):
    batch = [j.to(DEVICE) for j in i]
    output = model(batch[0].float(), batch[1], batch[2])

    # append to talley
    logits = np.append(logits, output["logits"].detach().cpu().numpy(), 0)
    labels = np.append(labels, i[2].numpy(), 0)

    torch.cuda.empty_cache()

try:
    prec_recc, roc, cm, acc = tensor_metrics(logits, labels)
    run.log({"val_prec_recc": prec_recc,
             "val_confusion": cm,
             "val_roc": roc,
             "val_acc": acc})
    # model.train()
except ValueError:
    breakpoint()

# Saving
print("Saving model...")
os.mkdir(f"./models/{run.name}")
torch.save(model, f"./models/{run.name}/model.save")
torch.save(optimizer, f"./models/{run.name}/optimizer.save")


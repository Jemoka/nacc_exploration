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

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random

# model
from model import NACCModel


CONFIG = {
    "fold": 0,
    "featureset": "combined",
    "task": "current"
}

ONE_SHOT = True
# ONE_SHOT = False
ONLINE = False
# ONLINE = TRUE

if ONE_SHOT:
    run = wandb.init(project="nacc", entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))
else:
    run = wandb.init(project="nacc-kfold", entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))

config = run.config

BATCH_SIZE = 128
EPOCHS = 64
LR = 0.0001
NHEAD = 8
NLAYERS = 8
HIDDEN = 2048
FOLD = config.fold
FEATURESET = config.featureset

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# loading data
class NACCCurrentDataset(Dataset):

    def __init__(self, file_path, feature_path,
              # skipping 2 impaired because of labeling inconsistency
                 target_feature="NACCUDSD", target_indicies=[1,3,4],
                 emph=3, fold=0):
        """The NeuralPsycology Dataset

        Arguments:

        file_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_feature] (str): the name of feature to serve as the target
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [emph] (int): the index to emphasize 
        [fold] (int): the n-th fold to select
        """

        # initialize superclass
        super(NACCCurrentDataset, self).__init__()

        # Read the raw dataset
        self.raw_data = pd.read_csv(file_path)

        # get the fature variables
        with open(feature_path, 'r') as df:
            lines = df.readlines()
            self.features = [i.strip() for i in lines]

        # skip elements whose target is not in the list
        self.raw_data = self.raw_data[self.raw_data[target_feature].isin(target_indicies)] 
        # Get a list of participants
        participants = self.raw_data["NACCID"]

        # Drop the parcticipants 
        self.raw_data = self.raw_data.drop(columns="NACCID")

        # Make it a multiindex by combining the experiment ID with the participant
        # so we can index by participant as first pass
        index_participant_correlated = list(zip(participants, pd.RangeIndex(0, len(self.raw_data))))
        index_multi = pd.MultiIndex.from_tuples(index_participant_correlated, names=["Participant ID", "Entry ID"])
        self.raw_data.index = index_multi

        # k fold
        participants = self.raw_data.index.get_level_values(0)

        kf = KFold(n_splits=10, random_state=7, shuffle=True)

        participant_set = pd.Series(list(set(participants)))
        splits = kf.split(participant_set)
        train_ids, test_ids = list(splits)[fold]

        train_participants = participant_set[train_ids]
        test_participants = participant_set[test_ids]

        # shuffle
        self.raw_data = self.raw_data.sample(frac=1)

        # Calculate the target data
        self.targets = self.raw_data[target_feature]
        self.data = self.raw_data[self.features] 

        # store the traget indicies
        self.__target_indicies = target_indicies

        # get number of features, by hoisting the get function up and getting length
        self._num_features = len(self.features)

        # crop the data for validatino
        self.val_data = self.data.loc[test_participants]
        self.val_targets = self.targets.loc[test_participants]

        self.data = self.data.loc[train_participants]
        self.targets = self.targets.loc[train_participants]

        # basic dataaug
        if emph:
            emph_features = self.targets==emph

            self.data = pd.concat([self.data, self.data[emph_features]])
            self.targets = pd.concat([self.targets, self.targets[emph_features]])

    def __process(self, data, target, index=None):
        # the discussed dataprep
        # if a data entry is <0 or >80, it is "not found"
        # so, we encode those values as 0 in the FEATURE
        # column, and encode another feature of "not-found"ness
        data_found = (data > 80) | (data < 0)
        data[data_found] = 0
        # then, the found-ness becomes a mask
        data_found_mask = (data_found)

        # if it is a sample with no tangible data
        # well give up and get another sample:
        if sum(~data_found_mask) == 0:
            if not index:
                raise ValueError("All-Zero found in validation!")
            indx = random.randint(2,5)
            if index-indx <= 0:
                return self[index+indx]
            else:
                return self[index-indx]
        
        # seed the one-hot vector
        one_hot_target = [0 for _ in range(len(self.__target_indicies))]
        # and set it
        one_hot_target[self.__target_indicies.index(target)] = 1

        return torch.tensor(data).long(), torch.tensor(data_found_mask).bool(), torch.tensor(one_hot_target).float()

    def __getitem__(self, index):
        # index the data
        data = self.data.iloc[index].copy()
        target = self.targets.iloc[index].copy()

        return self.__process(data, target, index)

    @functools.cache
    def val(self):
        """Return the validation set"""

        # collect dataset
        dataset = []

        print("Processing validation data...")

        # get it
        for index in tqdm(range(len(self.val_data))):
            try:
                dataset.append(self.__process(self.val_data.iloc[index].copy(),
                                            self.val_targets.iloc[index].copy()))
            except ValueError:
                continue # all zero ignore

        # return parts
        inp, mask, out = zip(*dataset)

        return torch.stack(inp).long(), torch.stack(mask).bool(), torch.stack(out).float()

    def __len__(self):
        return len(self.data)

dataset = NACCCurrentDataset("../investigator_nacc57.csv", f"../features/{FEATURESET}", fold=FOLD)

validation_set = TensorDataset(*dataset.val())
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = NACCModel(dataset._num_features, 3).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

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

model.train()
for epoch in range(EPOCHS):
    print(f"Currently training epoch {epoch}...")

    for i, batch in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
        # send batch to GPU if needed
        batch = [i.to(DEVICE) for i in batch]

        # run with actual backprop
        output = model(*batch)

        # backprop
        output["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        run.log({"loss": output["loss"].detach().cpu().item()})

model.eval()

# we track logits and labels and count them
# finally together eventually
logits = np.empty((0,3))
labels = np.empty((0,3))

print("Validating...")

# validation is large, so we do batches
for i in tqdm(iter(validation_loader)):
    output = model(*[j.to(DEVICE) for j in i])

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


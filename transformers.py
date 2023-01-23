# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd

import wandb

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import f1_score

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random

VALIDATE_EVERY = 20

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize the model
CONFIG = {
    "epochs": 128,
    "lr": 1e-4,
    "batch_size": 512,
}

# set up the run
# run = wandb.init(project="nacc", entity="jemoka", config=CONFIG)
run = wandb.init(project="nacc", entity="jemoka", config=CONFIG, mode="disabled")
config = run.config

BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LEARNING_RATE=config.lr

# loading data
class NACCNeuralPsychDataset(Dataset):

    def __init__(self, file_path, feature_path,
              # skipping 2 impaired because of labeling inconsistency
                 target_feature="NACCUDSD", target_indicies=[1,3,4]):
        """The NeuralPsycology Dataset

        Arguments:

        file_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_feature] (str): the name of feature to serve as the target
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        """

        # initialize superclass
        super(NACCNeuralPsychDataset, self).__init__()

        # Read the raw dataset
        self.raw_data = pd.read_csv(file_path)

        # get the fature variables
        with open(feature_path, 'r') as df:
            lines = df.readlines()
            self.features = [i.strip() for i in lines]

        # skip elements whose target is not in the list
        self.raw_data = self.raw_data[self.raw_data[target_feature].isin(target_indicies)] 

        # Calculate the target data
        self.targets = self.raw_data[target_feature]
        self.data = self.raw_data[self.features] 

        self._num_targets = len(self.features)

        # store the traget indicies
        self.__target_indicies = target_indicies

    def __getitem__(self, index):
        # index the data
        data = self.data.iloc[index]
        target = self.targets.iloc[index]

        # seed the one-hot vector
        one_hot_target = [0 for _ in range(len(self.__target_indicies))]
        # and set it
        one_hot_target[self.__target_indicies.index(target)] = 1

        return torch.tensor(data).float(), torch.tensor(one_hot_target).float()

    def __len__(self):
        return len(self.data)

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_features, num_classes, nhead=8, nlayers=6, hidden=256):
        # call early initializers
        super(NACCModel, self).__init__()

        # the entry network ("linear embedding")
        self.embedding = nn.Linear(num_features, hidden)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # prediction network
        self.linear = nn.Linear(hidden, num_classes)

        # util layers
        self.softmax = nn.Softmax(1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):

        net = self.embedding(x)
        net = self.encoder(net)
        net = self.linear(net)
        net = self.softmax(net)

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])

            self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }

dataset = NACCNeuralPsychDataset("./investigator_nacc57.csv", "./neuralpsych")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = NACCModel(dataset._num_targets, 4).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# calculate the f1 from tensors
tensor_f1 = lambda logits, labels: f1_score(torch.argmax(labels.cpu(), 1),
                                            torch.argmax(logits.detach().cpu(), 1),
                                            average='weighted')

model.train()
for epoch in range(EPOCHS):
    print(f"Currently training epoch {epoch}...")

    for i, batch in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
        # send batch to GPU if needed
        batch = [i.to(DEVICE) for i in batch]

        # generating validation output
        if i % VALIDATE_EVERY == 0:
            model.eval()
            output = model(*batch)
            run.log({"val_loss": output["loss"].detach().cpu().item(),
                    "val_f1": tensor_f1(output["logits"], batch[1])})
            model.train()
            continue

        # run with actual backprop
        labels = batch[1]
        output = model(*batch)

        # backprop
        output["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        run.log({"loss": output["loss"].detach().cpu().item(),
                "f1": tensor_f1(output["logits"], batch[1])})


# Saving
print("Saving model...")
os.mkdir(f"./models/{run.name}")
torch.save(model, f"./models/{run.name}/model.save")
torch.save(optimizer, f"./models/{run.name}/optimizer.save")



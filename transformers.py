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

from sklearn.metrics import precision_recall_fscore_support

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random

VALIDATE_EVERY = 20

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# initialize the model
CONFIG = {
    "epochs": 128,
    "lr": 1e-4,
    "batch_size": 128,
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
                 target_feature="NACCUDSD", target_indicies=[1,3,4],
                 emph=3, val=0.001):
        """The NeuralPsycology Dataset

        Arguments:

        file_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_feature] (str): the name of feature to serve as the target
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [emph] (int): the index to emphasize 
        [val] (float): number of samples to leave in the validation set
        """

        # initialize superclass
        super(NACCNeuralPsychDataset, self).__init__()

        # Read the raw dataset
        self.raw_data = pd.read_csv(file_path)
        # shuffle
        self.raw_data = self.raw_data.sample(frac=1)

        # get the fature variables
        with open(feature_path, 'r') as df:
            lines = df.readlines()
            self.features = [i.strip() for i in lines]

        # basic dataaug
        if emph:
            self.raw_data = pd.concat([self.raw_data, self.raw_data[self.raw_data[target_feature]==emph]])

        # skip elements whose target is not in the list
        self.raw_data = self.raw_data[self.raw_data[target_feature].isin(target_indicies)] 

        # Calculate the target data
        self.targets = self.raw_data[target_feature]
        self.data = self.raw_data[self.features] 

        # store the traget indicies
        self.__target_indicies = target_indicies

        # get number of features, by hoisting the get function up and getting length
        self._num_features = len(self.features)

        # crop the data for validatino
        val_count = int(len(self.data)*val)
        self.val_data = self.data.iloc[:val_count]
        self.val_targets = self.targets.iloc[:val_count]

        self.data = self.data.iloc[val_count:]
        self.targets = self.targets.iloc[val_count:]


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

    def val(self):
        """Return the validation set"""

        # collect dataset
        dataset = []

        # get it
        for index in range(len(self.val_data)):
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

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_features, num_classes, nhead=8, nlayers=6, hidden=256):
        # call early initializers
        super(NACCModel, self).__init__()

        # the entry network ("linear embedding")
        self.embedding = nn.Embedding(num_features, hidden)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # dropoutp!
        self.dropout = nn.Dropout(0.1)

        # prediction network
        self.linear = nn.Linear(num_features, num_classes)

        # util layers
        self.softmax = nn.Softmax(1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, mask, labels=None):

        net = self.embedding(x)
        # recall transformers are seq first
        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)
        net = self.dropout(net)
        net = torch.mean(net, dim=2)
        net = self.linear(net)
        net = self.softmax(net)

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }

dataset = NACCNeuralPsychDataset("./investigator_nacc57.csv", "./neuralpsych")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

VALIDATION_SET = [i.to(DEVICE) for i in dataset.val()]

model = NACCModel(dataset._num_features, 3).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# calculate the f1 from tensors
def tensor_metrics(logits, labels):
    label_indicies = torch.argmax(labels.cpu(), 1)
    logits_indicies  = logits.detach().cpu()

    class_names = ["Control", "MCI", "Dementia"]

    pr_curve = wandb.plot.pr_curve(label_indicies, logits_indicies, labels = class_names)
    roc = wandb.plot.roc_curve(label_indicies, logits_indicies, labels = class_names)
    cm = wandb.plot.confusion_matrix(
        y_true=np.array(label_indicies), # can't labels index by scalar tensor
        probs=logits_indicies,
        class_names=class_names
    )
    return pr_curve, roc, cm

model.train()
for epoch in range(EPOCHS):
    print(f"Currently training epoch {epoch}...")

    for i, batch in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
        # send batch to GPU if needed
        batch = [i.to(DEVICE) for i in batch]

        # generating validation output
        if i % VALIDATE_EVERY == 0:
            # model.eval()
            output = model(*VALIDATION_SET)
            try:
                prec_recc, roc, cm = tensor_metrics(output["logits"], VALIDATION_SET[2])
                run.log({"val_loss": output["loss"].detach().cpu().item(),
                            "val_prec_recc": prec_recc,
                            "val_confusion": cm,
                            "val_roc": roc})
                # model.train()
            except ValueError:
                breakpoint()

        # run with actual backprop
        output = model(*batch)

        # backprop
        output["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        run.log({"loss": output["loss"].detach().cpu().item()})

# Saving
print("Saving model...")
os.mkdir(f"./models/{run.name}")
torch.save(model, f"./models/{run.name}/model.save")
torch.save(optimizer, f"./models/{run.name}/optimizer.save")



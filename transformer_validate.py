# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd

from tqdm import tqdm
import random

import matplotlib

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import pathlib

import seaborn as sns

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# model to load
MODEL = "./models/eternal-yogurt-27"

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

        # flatten!
        self.flatten = nn.Flatten()

        # dropoutp!
        self.dropout = nn.Dropout(0.1)

        # prediction network
        self.linear1 = nn.Linear(hidden*num_features, hidden)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden, num_classes)
        self.softmax = nn.Softmax(1)

        # loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, mask, labels=None):

        net = self.embedding(x)
        # recall transformers are seq first
        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)
        net = self.dropout(net)
        net = self.flatten(net)
        net = self.linear1(net)
        net = self.gelu(net)
        net = self.linear2(net)
        net = self.dropout(net)
        net = self.softmax(net)

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }

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

# load data
dataset = NACCNeuralPsychDataset("./investigator_nacc57.csv", "./neuralpsych")

# load model
model = torch.load(os.path.join(MODEL, "model.save"),
                   map_location=DEVICE).to(DEVICE)

# elements
labels = []
confidences = []
results = []
feats_presents = []

for indx in tqdm(range(0, len(dataset), 100)):
    i = dataset[indx] 
    
    # pass through the model
    inp = [j.unsqueeze(0).to(DEVICE) for j in i]
    oup = model(*inp)

    # get stats
    label = torch.argmax(inp[2]).item()
    confidence = max(oup["logits"].squeeze()).cpu().item()
    result = (torch.argmax(oup["logits"].unsqueeze(0)) == label).cpu().item()
    feats_present = (sum(inp[1].squeeze())/len(inp[1].squeeze())).item()

    labels.append(label)
    confidences.append(confidence)
    results.append(result)
    feats_presents.append(feats_present)

df = pd.DataFrame({"label":labels,
                   "correct":results,
                   "confidence": confidences,
                   "feature_percent": feats_presents})

# df.to_csv(f"./models/{pathlib.Path(MODEL).stem}.csv")

### Accuracy Breakdown ###
# Control
control = df[df.label == 0]
control_acc = sum(control.correct)/len(control)
# MCI
mci = df[df.label == 1]
mci_acc = sum(mci.correct)/len(mci)
# Dementia
dementia = df[df.label == 2]
dementia_acc = sum(dementia.correct)/len(dementia)

df.groupby(round(df.feature_percent, 1)).mean()


def read_attention(mod, inp, out):
    return out[0].sum(-1).squeeze()

hook = model.encoder.layers[-1].self_attn.register_forward_hook(read_attention)

i = dataset[0] 

# pass through the model
inp = [j.unsqueeze(0).to(DEVICE) for j in i]
oup = model(*inp)

hook.remove()

tmp = pd.Series([-1.3668, -1.8046, -1.7830, -0.8079, -1.8558, -1.7853, -1.2525, -2.0575,
                 -1.7061, -1.7932, -2.4580, -1.5750, -0.7206, -2.3476, -1.8048, -2.1570,
                 -1.9656, -0.1466, -1.6438, -1.0996, -1.6864,  1.8741, -1.4301, -2.0249,
                 0.8634, -1.5356, -0.8287, -0.0786, -2.6008, -1.6228, -0.6838, -1.0613,
                 -1.7739, -2.4485, -1.1399, -2.2909, -0.6110, -1.2949, -2.5429, -1.8271,
                 -1.4359, -1.3885, -2.0300, -1.1423,  0.8273, -3.2975, -2.3298, -1.0742,
                 -1.9183, -0.5756, -2.1312, -1.9953, -1.6369, -2.0601, -1.8739, -2.2016,
                 -1.6294, -2.3776, -2.3291, -1.8196, -1.5457, -1.0589, -2.2160, -0.3811,
                 -0.7995, -0.7835, -1.7580, -0.9875, -1.6470, -2.4313, -1.9902, -1.1369,
                 -1.0913, -1.3479, -1.4603, -2.2017, -1.4282, -0.8007, -1.3865, -1.7188,
                 -1.3541, -1.5356, -2.4453, -0.9674, -1.9275, -1.8818, -1.7373, -1.6338,
                 -0.8339, -0.9110, -0.5940, -2.1722, -0.7534, -1.6531, -2.5053, -2.0346,
                 -1.7322, -1.4252, -1.1248, -1.4856, -1.2738, -1.6907, -1.4741])

tmp = ((tmp-tmp.mean())/tmp.std()).tolist()

zipped_attention = list(zip([i.item() for i in inp[0].squeeze()], tmp))
zipped_attention


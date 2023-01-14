# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import tokenizers
import numpy as np
import pandas as pd

import wandb

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Ling utilities
import nltk
from nltk import sent_tokenize

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
    "epochs": 3,
    "lr": 3e-3,
    "batch_size": 32,
}

# set up the run
# run = wandb.init(project="mutembeds", entity="jemoka", config=CONFIG)
run = wandb.init(project="nacc", entity="jemoka", config=CONFIG, mode="disabled")
config = run.config

BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LEARNING_RATE=config.lr

# loading data
class NACCNeuralPsychDataset(Dataset):

    def __init__(self, file_path, feature_path,
                 target_feature="NACCUDSD", target_indicies=[1,2,3,4]):
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

        # Calculate the target data
        self.targets = self.raw_data[target_feature]
        self.data = self.raw_data[self.features] 

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

dataset = NACCNeuralPsychDataset("./investigator_nacc57.csv", "./neuralpsych")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
data_iter = iter(dataloader)





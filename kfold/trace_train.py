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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

with open("../attention_analysis_data_bal.bin", 'rb') as df:
    data = torch.load(df)

(X,y) = data["train"]
(X_val,y_val) = data["test"]

X, y = unison_shuffled_copies(X, y)

sum(torch.argmax(y_val, dim=1)==0)
sum(torch.argmax(y_val, dim=1)==1)
y_val.shape

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib.pyplot as plt

tree = DecisionTreeClassifier()
tree = tree.fit(X, y)
tree.score(X_val, y_val) 
# plt.figure(figsize=(300,100), dpi=100).clear()
# plot = plot_tree(tree, fontsize=20)
# plt.savefig(f"./tree.png")

forest = RandomForestClassifier()
forest = forest.fit(X, y)
forest.score(X_val, y_val)

bayes = GaussianNB()
bayes = bayes.fit(X, y)
bayes.score(X_val, y_val)

logreg = LogisticRegression()
logreg = logreg.fit(X, y)
logreg.score(X_val, y_val)

svc = SVC()
svc = svc.fit(X, y)
svc.score(X_val, y_val)


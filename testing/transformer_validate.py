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
sns.set_theme("talk", style="darkgrid", palette="crest")



# initialize the device
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model to load
MODEL = "../models/vulcan-bird-of-prey-39"

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_features, num_classes, nhead=8, nlayers=6, hidden=256):
        # call early initializers
        super(NACCModel, self).__init__()

        # the entry network ("linear embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
        self.embedding = nn.Embedding(81, hidden)
        
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

    def forward(self, x, mask=None, labels=None):

        net = self.embedding(x)
        # recall transformers are seq first
        if mask == None:
            net = self.encoder(net.transpose(0,1)).transpose(0,1)
        else:
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
dataset = NACCNeuralPsychDataset("../investigator_nacc57.csv", "../features/combined")

data = dataset.data

dataset.targets.value_counts()

# I'm sorry
CHECK = lambda i: "Dementia" if i == 4 else "Control" if i == 1 else "MCI"

percentage_missing = (data==-4).apply(sum, axis=1)/(data==-4).apply(len, axis=1)
missing_data = pd.DataFrame({"Percent of Data Available": 1-percentage_missing,
                             "Target": dataset.targets.apply(CHECK)})

plt.figure().clear()
plt.axes().set_box_aspect(1)
sns.boxplot(data=missing_data, x="Target", y="Percent of Data Available").set(title="Percentage of Input Features Available for a Given Sample")
# sns.heatmap(normal_norm, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), annot=True,
#             xticklabels=["Control", "MCI", "Dementia"],
#             yticklabels=["Control", "MCI", "Dementia"]).set(title="Clinical Diagnosis Prediction, 5% Holdout Validation Confusion Matrix (86.9% Overall Acc.)")
plt.show()




# # load model
model = torch.load(os.path.join(MODEL, "model.save"),
                   map_location=DEVICE).to(DEVICE)

from torch.onnx import export 

inp = [j.unsqueeze(0).to(DEVICE) for j in dataset[121]]


dataset[0]

from torchviz import make_dot
viz = make_dot(model(*inp)["loss"], params=dict(model.named_parameters()))
viz.save("../figures/model.dot")

export(model, inp[0], os.path.join(MODEL, "model.onnx"))

# # elements
# labels = []
# confidences = []
# results = []
# feats_presents = []

# for indx in tqdm(range(0, len(dataset), 100)):
#     i = dataset[indx] 
    
#     # pass through the model
#     inp = [j.unsqueeze(0).to(DEVICE) for j in i]
#     oup = model(*inp)

#     # get stats
#     label = torch.argmax(inp[2]).item()
#     confidence = max(oup["logits"].squeeze()).cpu().item()
#     result = (torch.argmax(oup["logits"].unsqueeze(0)) == label).cpu().item()
#     feats_present = (sum(inp[1].squeeze())/len(inp[1].squeeze())).item()

#     labels.append(label)
#     confidences.append(confidence)
#     results.append(result)
#     feats_presents.append(feats_present)

# df = pd.DataFrame({"label":labels,
#                    "correct":results,
#                    "confidence": confidences,
#                    "feature_percent": feats_presents})

# # df.to_csv(f"./models/{pathlib.Path(MODEL).stem}.csv")

# ### Accuracy Breakdown ###
# # Control
# control = df[df.label == 0]
# control_acc = sum(control.correct)/len(control)
# # MCI
# mci = df[df.label == 1]
# mci_acc = sum(mci.correct)/len(mci)
# # Dementia
# dementia = df[df.label == 2]
# dementia_acc = sum(dementia.correct)/len(dementia)

# df.groupby(round(df.feature_percent, 1)).mean()

# def read_attention(mod, inp, out):
#     print(out)
#     return out

# tmp = pd.Series([-0.6017, -1.4213, -0.1744, -0.7487,  3.9498,  5.5751, -4.4889, -2.1936,
#        -3.6340,  0.1321,  1.3879,  3.3084,  5.8068, -0.6585, -1.5083,  0.8317,
#        0.8723,  3.7855,  4.4545,  4.4564, -1.7827, -1.1391, -5.1701, -3.4897,
#        -1.2304, -3.1686, -2.0418,  2.1061,  7.9796, -0.7690,  1.9104, -5.3666,
#        0.4209, -2.5557, -1.9161, -2.4140, -0.4391, -1.0187, -1.9612, -2.4329,
#        5.6491,  5.4506,  4.3944, -0.5510,  3.0210, -5.3870,  5.9335, -1.9257,
#        6.9999,  5.5075, -5.2941,  3.9323,  0.0622, -2.2690,  5.3083,  2.5578,
#        3.0858,  1.7683,  5.7945,  5.2832,  5.2984,  2.9409, -0.5184,  8.4684,
#        4.3989,  3.9700,  3.7639,  0.4790,  8.1474,  1.1683, -0.9344, -6.2655,
#        -1.6892, -0.4481, -0.5459, -2.6126, -1.8289,  1.8540,  3.0106, -1.8860,
#        -1.1338, -2.3526, -2.2765, -3.0212, -2.4682, -2.4467, -2.2888, -3.1426,
#        -2.6542, -2.0798, -1.5492, -1.1480, -2.1481, -4.5286, -1.4182, -0.4367,
#        -2.1292, -3.6011, -3.7934, -2.4401, -2.5840, -2.7806, -1.9196])



# hook = model.encoder.layers[0].self_attn.register_forward_hook(read_attention)

# i = dataset[30] 

# # pass through the model
# inp = [j.unsqueeze(0).to(DEVICE) for j in i]
# oup = model(*inp)

# hook.remove()



# with open("./neuralpsych", 'r') as df:
#     lines = df.readlines()
#     features = [i.strip() for i in lines]

# # features
# tmp = tmp*(1-(i[1].float())).numpy()
# tmp


# zipped_attention = list(zip(features, [i.item() for i in inp[0].squeeze()], tmp))

# tmp2 = pd.DataFrame(zipped_attention)

# print(tmp2.to_string())

# i[2]

# tmp2.columns=["feature", "data", "attention"]
# tmp2[tmp2["attention"]==0].loc[:,"attention"] = 0
# tmp2
# tmp2.to_csv("attention_control.csv")
# tmp2[tmp2["attention"] == 0]

# i[2]


# zipped_attention

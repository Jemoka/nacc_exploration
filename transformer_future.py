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

from tqdm.auto import tqdm

import wandb

tqdm.pandas()

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

VALIDATE_EVERY = 20

# initialize the model
CONFIG = {
    "epochs": 128,
    "lr": 5e-5,
    "batch_size": 128,
    "hidden": 256,
    "heads": 8,
    "encoder_layers": 6,
    "model": "eternal-yogurt-27",
    "bound": (1,3)
}

# set up the run
# run = wandb.init(project="nacc_future", entity="jemoka", config=CONFIG)
run = wandb.init(project="nacc_future", entity="jemoka", config=CONFIG, mode="disabled")
config = run.config

BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LEARNING_RATE=config.lr
HIDDEN = config.hidden
HEADS = config.heads
LAYERS = config.encoder_layers
MODEL = f"./models/{config.model}"
BOUND = config.bound

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
                 val=0.001, bound=BOUND):
        """The NeuralPsycology Dataset

        Arguments:

        file_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_feature] (str): the name of feature to serve as the target
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [val] (float): number of samples to leave in the validation set
        [bound] ((int, int)): how many years starting/ending to crop the range
        """

        # initialize superclass
        super(NACCNeuralPsychDataset, self).__init__()

        # Read the raw dataset
        raw_data = pd.read_csv(file_path)

        # get the fature variables
        with open(feature_path, 'r') as df:
            lines = df.readlines()
            self.features = [i.strip() for i in lines]

        # skip elements whose target is not in the list
        raw_data = raw_data[raw_data[target_feature].isin(target_indicies)] 
        # Get a list of participants
        participants = raw_data["NACCID"]

        # Drop the parcticipants 
        raw_data = raw_data.drop(columns="NACCID")

        # Make it a multiindex by combining the experiment ID with the participant
        # so we can index by participant as first pass
        index_participant_correlated = list(zip(participants, pd.RangeIndex(0, len(raw_data))))
        index_multi = pd.MultiIndex.from_tuples(index_participant_correlated, names=["Participant ID", "Entry ID"])
        raw_data.index = index_multi

        # cosntruct a "diagnosis-in-nyears" column
        age_date_data = raw_data[["NACCUDSD", "NACCAGE"]]

        max_age_plus_fifty = max(raw_data.NACCAGE)+50

        def find_data(grp):
            """find the age til dementia

            grp: DataFrameGroupBy 
            """

            # get dementia indicies
            mci_indicies = grp[(grp.NACCUDSD==3)]
            dementia_indicies = grp[(grp.NACCUDSD==4)]

            # store demented
            ultimate_diag_type = (3 if(len(mci_indicies)) else 1) if (len(dementia_indicies) == 0) else 4

            # if length is 0 "the person never had dementia"
            if ultimate_diag_type == 1:
                # make the dementia age max+50 i.e. they will get dementia a long time later
                initial_dementia_age = max_age_plus_fifty
            elif ultimate_diag_type == 3:
                # get the first MCI age 
                initial_dementia_age = mci_indicies.iloc[0].NACCAGE
            else:
                # get the first dEMENITA age 
                initial_dementia_age = dementia_indicies.iloc[0].NACCAGE

            # get the number of years to the age of dementia
            age_til_dementia = (initial_dementia_age - grp.NACCAGE).apply(lambda x:0 if x<0 else x)

            # will be demented will be repeted n times to fit length
            return age_til_dementia, [ ultimate_diag_type for _ in range(len(grp)) ]

        # weird data gymnastics to unpeel the two columns
        print("Loading timeseries groups...")
        age_til_dementia, ultimate_diag_type = zip(*(age_date_data.groupby(level=0,axis=0).progress_apply(find_data)))
        print("Done!")

        # create series for the outputs
        age_til_dementia_series = pd.concat(age_til_dementia)
        ultimate_diag_series = pd.Series([j for i in ultimate_diag_type for j in i])

        # align the index
        ultimate_diag_series.index = age_til_dementia_series.index

        # dump in result
        raw_data.loc[:,"age_til_dementia"] = age_til_dementia_series.sort_index(level=1)
        raw_data.loc[:, "ultimate_diag_type"] = ultimate_diag_series.sort_index(level=1)

        # shuffle
        raw_data_sample = raw_data.sample(frac=1)
        age_til_dementia_sample = raw_data_sample["age_til_dementia"]
        raw_data_sample = raw_data_sample[((age_til_dementia_sample>bound[0])&(age_til_dementia_sample<bound[1]))|(age_til_dementia_sample > 50)] # > 50 is control 

        # shuffle again and sample based on median size
        median_count = raw_data_sample.ultimate_diag_type.value_counts().median()
        raw_data_sample = raw_data_sample.groupby("ultimate_diag_type").sample(n=int(median_count),
                                                                            replace=True)

        # shuffle again and sample based on median size
        raw_data_sample =  raw_data_sample.sample(frac=1)

        # set
        self.raw_data = raw_data_sample

        # Calculate the target data. this is not trivial.
        self.targets = raw_data_sample["ultimate_diag_type"]

        # then isolate the daata
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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

VALIDATION_SET = [i.to(DEVICE) for i in dataset.val()]

# load model
model = torch.load(os.path.join(MODEL, "model.save"),
                   map_location=DEVICE).to(DEVICE)
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

    acc = sum(label_indicies == torch.argmax(logits_indicies, dim=1))/len(label_indicies)

    return pr_curve, roc, cm, acc

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
                prec_recc, roc, cm, acc = tensor_metrics(output["logits"], VALIDATION_SET[2])
                run.log({"val_loss": output["loss"].detach().cpu().item(),
                            "val_prec_recc": prec_recc,
                            "val_confusion": cm,
                            "val_roc": roc,
                         "val_acc": acc})
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
os.mkdir(f"./models/F_{config.model}_{run.name}")
torch.save(model, f"./models/F_{config.model}_{run.name}/model.save")
torch.save(optimizer, f"./models/F_{config.model}_{run.name}/optimizer.save")


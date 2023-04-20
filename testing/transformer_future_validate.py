# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd

from collections import defaultdict

from tqdm import tqdm
import random

import matplotlib

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import pathlib

import seaborn as sns
sns.set_theme("talk", style="darkgrid", palette="crest")

import matplotlib.patches as patches
tqdm.pandas()

# initialize the device
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model to load
MODEL = "../models/F_vulcan-bird-of-prey-39_light-aardvark-7"

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_features, num_classes, nhead=8, nlayers=6, hidden=256):
        # call early initializers
        super(NACCModel, self).__init__()

        # the entry network ("linear embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
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
                 val=0.01, bound=(1,3)):
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
dataset = NACCNeuralPsychDataset("../investigator_nacc57.csv", "../features/combined")

# load model
model = torch.load(os.path.join(MODEL, "model.save"),
                   map_location=DEVICE).to(DEVICE)

dataset[0]
model

inp = [j.unsqueeze(0).to(DEVICE) for j in dataset[121]]
oup = model(*inp)
inp

plt.figure().clear()
plt.axes().set_box_aspect(1)

# poking around embedding
torch.dist(model.embedding(torch.tensor(5)),
           model.embedding(torch.tensor(30)))

embedding_values = []
for i in range(0, 80):
    embedding_values.append(model.embedding(torch.tensor(i)).detach().numpy())

embedding_values = np.array(embedding_values)
embedding_values.shape

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
transformed = pca.fit_transform(embedding_values)

sns.scatterplot(x=transformed[:30,0],
                y=transformed[:30,1],
                hue=range(30))
plt.show()

# poking around attention activation
sample = inp[0]
sample_embedded = model.embedding(sample)

attn_out, attn_weights = model.encoder.layers[0].self_attn(sample_embedded,
                                                           sample_embedded,
                                                           sample_embedded)
attn_weights_squeezed = attn_weights.squeeze().squeeze()
attn_weights_masked = torch.masked_fill(attn_weights_squeezed, 
                                        inp[1], 0).squeeze()
attention_matrix = torch.outer(attn_weights_masked,
                               attn_weights_masked)

labels = ["Patient History" for _ in range(69)] + ["Neuro-Psychological Battery" for _ in range(103)]
features = dataset.features

attention_data = pd.DataFrame({ "attention": attn_weights_masked.detach(),
                                "label": labels,
                                "feature": features,
                                "masked": inp[1].squeeze()})
# voilen activations by feature
plt.figure().clear()
plt.axes().set_box_aspect(1)
sns.violinplot(data=attention_data[~attention_data.masked],
               y="attention",
               x="label").set(title="Attention Activations for Dementia Sample",
                              xlabel="Feature Type",
                              ylabel="Attention Activation")
plt.show()

# cross attention activation
# tool to unflatten INDICIES to back to get the value
def get_indicies(index, length=172):
    return index // length, index % length, 

cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()
_, indicies = torch.topk(torch.abs(torch.flatten(cross_attn)), 50)
indicies
attention_indicies = [get_indicies(i.item()) for i in list(indicies)]
attention_features = [(dataset.features[i],
                       dataset.features[j]) for i,j in
                      attention_indicies]
attention_indicies
attention_features

def contains_feat_type(index, length=172):
    history = False
    battery = False

    if (index // length) < 69 or (index % length) < 69:
        history = True
    if (index // length) >= 69 or (index % length) >= 69:
        battery = True

    return history, battery

cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()
_, indicies = torch.topk(torch.abs(torch.flatten(cross_attn)), 50)
feat_types = [contains_feat_type(i.item()) for i in list(indicies)]

# cross attention activation percentage

activation_perc_data = []
apperance_counter = defaultdict(int)

for i in tqdm(range(len(dataset))):
    inp = [j.unsqueeze(0).to(DEVICE) for j in dataset[i]]

    sample = inp[0]
    sample_embedded = model.embedding(sample)

    attn_out, attn_weights = model.encoder.layers[0].self_attn(sample_embedded,
                                                            sample_embedded,
                                                            sample_embedded)
    def contains_feat_type(index, length=172):
        history = False
        battery = False

        if (index // length) < 69 or (index % length) < 69:
            history = True
        if (index // length) >= 69 or (index % length) >= 69:
            battery = True

        apperance_counter[index // length] += 1
        apperance_counter[index % length] += 1

        return history, battery

    cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()
    _, indicies = torch.topk(torch.abs(torch.flatten(cross_attn)), 17)
    feat_types = [contains_feat_type(i.item()) for i in list(indicies)]

    # get percentage containing history
    containing_history = sum([i[0] for i in feat_types]) / len(feat_types)
    containing_battery = sum([i[1] for i in feat_types]) / len(feat_types)

    if inp[2][0][0] == 1:
        label = "Control"
    elif inp[2][0][1] == 1:
        label = "MCI"
    elif inp[2][0][2] == 1:
        label = "Dementia"

    activation_perc_data.append((containing_history,
                                 containing_battery,
                                 label))
apperance_counter = dict(apperance_counter)
common = sorted(list(apperance_counter.items()), key=lambda x:x[1], reverse=True)
common_df = pd.DataFrame(common)
common_df.columns = ["featid", "Occurrence"]
common_df["Type"] = common_df.featid.apply(lambda x:"Neuro-Psychological Battery"
                                           if x >= 69 else "Patient History")
common_df["Feature Name"] = common_df.featid.apply(lambda x:dataset.features[x])
common_df.drop(columns=["featid"], inplace=True)
common_df[:10]



activation_df = pd.DataFrame(activation_perc_data)
activation_df

activation_df.columns=["Patient History",
                       "Neuro-Psychological Battery",
                       "Outcome"]

from scipy.stats import ttest_ind
activation_df.groupby("Outcome")["Neuro-Psychological Battery"].median()

C = activation_df[activation_df.Outcome=="Control"]["Patient History"]
D = activation_df[activation_df.Outcome=="Dementia"]["Patient History"]
M = activation_df[activation_df.Outcome=="Mci"]["Patient History"]

ttest_ind(C, D)
ttest_ind(C, M)
ttest_ind(D, M)


plt.figure().clear()
plt.axes().set_box_aspect(1)
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.violinplot(data=activation_df, x="Outcome", y="Patient History", ax=ax1)
sns.violinplot(data=activation_df, x="Outcome", y="Neuro-Psychological Battery", ax=ax2)
ax1.set(ylim=(-0.2,1.2))
ax2.set(ylim=(-0.2,1.2))
plt.suptitle("Appearance in Top-10% Most Attended-To Features")
plt.show()

activation_df

indicies.shape
cross_attn

plt.figure().clear()
plt.axes().set_box_aspect(1)
ax = sns.heatmap(data=cross_attn,
                 cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 center=0, vmin=-250, vmax=250)
ax.set(title="First Layer Cross-Attention for a Single MCI Sample",
                                               xlabel=None,
                                               ylabel=None)
# p1 = patches.Rectangle((1,1), 69,69,linewidth=2, edgecolor="g", facecolor="none")
# p2 = patches.Rectangle((1,1), 69,170,linewidth=2, edgecolor="b", facecolor="none")
# p3 = patches.Rectangle((1,1), 170,69,linewidth=2, edgecolor="b", facecolor="none")
# ax.add_patch(p2)
# ax.add_patch(p3)
# ax.add_patch(p1)
plt.show()



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

from datasets import *

####### MODEL OPS #######

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

# model to load
MODEL = "../models/atomic-valley-21"

# load data
dataset = NACCFutureDataset("../investigator_nacc57.csv", "../features/combined")

# # load model
model = torch.load(os.path.join(MODEL, "model.save"),
                   map_location=DEVICE).to(DEVICE)

model.eval()

# get a single sample to poke around in
inp = [j.unsqueeze(0).to(DEVICE) for j in dataset.val()[21]]
oup = model(*inp)["logits"]
target = inp[-1]
confidence = oup[0][torch.argmax(oup[0])].item()

# if fold=0, seed=7, # 21 /should/ be an MCI sample

#########################

####### PARAMETER COUNT #######

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

PARAMETER_COUNT = get_n_params(model)

#########################

####### ATTENTION_ACTIVATION #######


# poking around attention activation
sample_embedded = model.embedding(inp[0])

attn_out, attn_weights = model.encoder.layers[0].self_attn(sample_embedded,
                                                           sample_embedded,
                                                           sample_embedded)
def get_indicies(index, length=172):
    return index // length, index % length, 

cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()
ax = sns.heatmap(data=cross_attn.cpu(),
                 cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 center=0, vmin=-250, vmax=250)
ax.set(title="First Layer Cross-Attention for a Single MCI Sample",
                                               xlabel=None,
                                               ylabel=None)
plt.show()

torch.argmax(cross_attn)

attn_out

# labels = ["Patient History" for _ in range(69)] + ["Neuro-Psychological Battery" for _ in range(103)]
# features = dataset.features

# attention_data = pd.DataFrame({ "attention": attn_weights_masked.detach(),
#                                 "label": labels,
#                                 "feature": features,
#                                 "masked": inp[1].squeeze()})
# # voilen activations by feature
# plt.figure().clear()
# plt.axes().set_box_aspect(1)
# sns.violinplot(data=attention_data[~attention_data.masked],
#                y="attention",
#                x="label").set(title="Attention Activations for Dementia Sample",
#                               xlabel="Feature Type",
#                               ylabel="Attention Activation")
# plt.show()

# # cross attention activation
# # tool to unflatten INDICIES to back to get the value
# def get_indicies(index, length=172):
#     return index // length, index % length, 

# cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()
# _, indicies = torch.topk(torch.abs(torch.flatten(cross_attn)), 50)
# indicies
# attention_indicies = [get_indicies(i.item()) for i in list(indicies)]
# attention_features = [(dataset.features[i],
#                        dataset.features[j]) for i,j in
#                       attention_indicies]
# attention_indicies
# attention_features

# def contains_feat_type(index, length=172):
#     history = False
#     battery = False

#     if (index // length) < 69 or (index % length) < 69:
#         history = True
#     if (index // length) >= 69 or (index % length) >= 69:
#         battery = True

#     return history, battery

# cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()
# _, indicies = torch.topk(torch.abs(torch.flatten(cross_attn)), 50)
# feat_types = [contains_feat_type(i.item()) for i in list(indicies)]

# # cross attention activation percentage

# activation_perc_data = []
# apperance_counter = defaultdict(int)

# for i in tqdm(range(len(dataset))):
#     inp = [j.unsqueeze(0).to(DEVICE) for j in dataset[i]]

#     sample = inp[0]
#     sample_embedded = model.embedding(sample)

#     attn_out, attn_weights = model.encoder.layers[0].self_attn(sample_embedded,
#                                                             sample_embedded,
#                                                             sample_embedded)
#     def contains_feat_type(index, length=172):
#         history = False
#         battery = False

#         if (index // length) < 69 or (index % length) < 69:
#             history = True
#         if (index // length) >= 69 or (index % length) >= 69:
#             battery = True

#         apperance_counter[index // length] += 1
#         apperance_counter[index % length] += 1

#         return history, battery

#     cross_attn = (attn_out @ attn_out.transpose(-2,-1)).squeeze(0).detach()
#     _, indicies = torch.topk(torch.abs(torch.flatten(cross_attn)), 17)
#     feat_types = [contains_feat_type(i.item()) for i in list(indicies)]

#     # get percentage containing history
#     containing_history = sum([i[0] for i in feat_types]) / len(feat_types)
#     containing_battery = sum([i[1] for i in feat_types]) / len(feat_types)

#     if inp[2][0][0] == 1:
#         label = "Control"
#     elif inp[2][0][1] == 1:
#         label = "MCI"
#     elif inp[2][0][2] == 1:
#         label = "Dementia"

#     activation_perc_data.append((containing_history,
#                                  containing_battery,
#                                  label))
# apperance_counter = dict(apperance_counter)
# common = sorted(list(apperance_counter.items()), key=lambda x:x[1], reverse=True)
# common_df = pd.DataFrame(common)
# common_df.columns = ["featid", "Occurrence"]
# common_df["Type"] = common_df.featid.apply(lambda x:"Neuro-Psychological Battery"
#                                            if x >= 69 else "Patient History")
# common_df["Feature Name"] = common_df.featid.apply(lambda x:dataset.features[x])
# common_df.drop(columns=["featid"], inplace=True)
# common_df[:10]

# common_df["Occurrence Percentage"] = common_df["Occurrence"] / sum(common_df["Occurrence"])
# # sum(common_df["Occurrence Percentage"][:10])

# battery_count = len(common_df[common_df.Type=="Neuro-Psychological Battery"])
# history_count = len(common_df[common_df.Type=="Patient History"])

# battery_mean = common_df[common_df.Type=="Neuro-Psychological Battery"]["Occurrence"].mean()
# history_mean = common_df[common_df.Type=="Patient History"]["Occurrence"].mean()

# plt.figure().clear()
# plt.axes().set_box_aspect(1)
# sns.boxenplot(data=common_df,
#               x="Type", y="Occurrence").set(title="Distribution of Occurrence Count of Each Top-10% Most Attended-To Features Across Dataset")
# plt.show()


# activation_df = pd.DataFrame(activation_perc_data)
# activation_df

# activation_df.columns=["Patient History",
#                        "Neuro-Psychological Battery",
#                        "Outcome"]

# from scipy.stats import ttest_ind
# activation_df.groupby("Outcome")["Neuro-Psychological Battery"].median()

# C = activation_df[activation_df.Outcome=="Control"]["Patient History"]
# D = activation_df[activation_df.Outcome=="Dementia"]["Patient History"]
# M = activation_df[activation_df.Outcome=="Mci"]["Patient History"]

# ttest_ind(C, D)
# ttest_ind(C, M)
# ttest_ind(D, M)


# plt.figure().clear()
# plt.axes().set_box_aspect(1)
# fig, (ax1, ax2) = plt.subplots(1, 2)
# sns.violinplot(data=activation_df, x="Outcome", y="Patient History", ax=ax1)
# sns.violinplot(data=activation_df, x="Outcome", y="Neuro-Psychological Battery", ax=ax2)
# ax1.set(ylim=(-0.2,1.2))
# ax2.set(ylim=(-0.2,1.2))
# plt.suptitle("Appearance Percentage in Top-10% Most Attended-To Features Across Dataset")
# plt.show()

# activation_df

# indicies.shape
# cross_attn

# plt.figure().clear()
# plt.axes().set_box_aspect(1)
# ax = sns.heatmap(data=cross_attn,
#                  cmap=sns.diverging_palette(220, 20, as_cmap=True),
#                  center=0, vmin=-250, vmax=250)
# ax.set(title="First Layer Cross-Attention for a Single MCI Sample",
#                                                xlabel=None,
#                                                ylabel=None)
# # p1 = patches.Rectangle((1,1), 69,69,linewidth=2, edgecolor="g", facecolor="none")
# # p2 = patches.Rectangle((1,1), 69,170,linewidth=2, edgecolor="b", facecolor="none")
# # p3 = patches.Rectangle((1,1), 170,69,linewidth=2, edgecolor="b", facecolor="none")
# # ax.add_patch(p2)
# # ax.add_patch(p3)
# # ax.add_patch(p1)
# plt.show()

# confusion_future = pd.read_csv("../figures/future_prediction.csv") 
# confusion_future

# combined = confusion_future[confusion_future.name=="light-aardvark-7"]
# neuralpsych = confusion_future[confusion_future.name=="sparkling-puddle-4"]

# combined_matrix = combined["nPredictions"].to_numpy().reshape(3,3)
# np_matrix = neuralpsych["nPredictions"].to_numpy().reshape(3,3)

# #norm
# combined_norm  = combined_matrix / np.sum(combined_matrix, axis=1)
# np_norm  = np_matrix / np.sum(np_matrix, axis=1)

# plt.figure().clear()
# fig, (ax1, ax2) = plt.subplots(1, 2)
# sns.heatmap(combined_norm, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), annot=True,
#             xticklabels=["Control", "MCI", "Dementia"],
#             yticklabels=["Control", "MCI", "Dementia"], ax=ax1).set(title="History + N-P Battery (87.5% Overall Acc.)")
# sns.heatmap(np_norm, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), annot=True,
#             xticklabels=["Control", "MCI", "Dementia"],
#             yticklabels=["Control", "MCI", "Dementia"], ax=ax2).set(title="N-P Battery Only (80.8% Overall Acc.)")
# fig.suptitle("Prognosis in 1-3 Years, 5% Holdout Validation Confusion Matrix")
# plt.show()


# confusion_normal = pd.read_csv("../figures/normal_prediction.csv") 
# confusion_normal

# normal_matrix = confusion_normal["nPredictions"].to_numpy().reshape(3,3)
# #norm
# normal_norm  = normal_matrix / np.sum(normal_matrix, axis=1)

# plt.figure().clear()
# plt.axes().set_box_aspect(1)
# sns.heatmap(normal_norm, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), annot=True,
#             xticklabels=["Control", "MCI", "Dementia"],
#             yticklabels=["Control", "MCI", "Dementia"]).set(title="Clinical Diagnosis Prediction, 5% Holdout Validation Confusion Matrix (86.9% Overall Acc.)")
# plt.show()



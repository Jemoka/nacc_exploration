import pandas

import pandas as pd
from pandas.arrays import BooleanArray
# type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# feature selection
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2
# type: ignore
# stats
from scipy.stats import kstest, pearsonr, f_oneway, chi2_contingency
from scipy import sparse
from tqdm import tqdm
# sns
import seaborn as sns
import numpy as np

import random

from matplotlib import pyplot as plt

# pandas tqdm
tqdm.pandas()

# Read the raw data
data = pd.read_csv("./investigator_nacc57.csv")

#######################################################################

# Create multiindex

# Get a list of participants
participants = data["NACCID"]

# Drop the parcticipants 
data = data.drop(columns="NACCID")

# Make it a multiindex by combining the experiment ID with the participant
# so we can index by participant as first pass
index_participant_correlated = list(zip(participants, pd.RangeIndex(0, len(data))))
index_multi = pd.MultiIndex.from_tuples(index_participant_correlated, names=["Participant ID", "Entry ID"])
data.index = index_multi

########################################################################

# output keys
output_key = data["NACCUDSD"]
# selection keys for MCI and WNL
IMP_key = output_key == 2
MCI_key = output_key == 3
DEM_key = output_key == 4
CTL_key = output_key == 1

########################################################################

# MoCA, MMSE (measure of global cognitive status)
# BNT, MINT (measure of word finding abilities)
# F-word generation, L-word Generation (measures of verbal fluency). Alternatively, could just use F+L word generation if more sites give this.
# Story Unit Recall + Logical Memory II (measure of delayed story memory)

# corrected features to match together
MOCA = "NACCMOCA"; MMSE = "NACCMMSE"
BOSTON = "BOSTON"; MINT = "MINTTOTS"
CRAFTSTORY = "CRAFTURS"; LOGICALMEM = "MEMUNITS"

all_features = [MOCA, MMSE, BOSTON,
                MINT, CRAFTSTORY, LOGICALMEM]

# Now, this list of features aren't exactly all available
# So, we will combine the top values that exist and normalize
# results pairwise.

is_missing = data[all_features]==-4
# calculate existance choorlation
is_missing.corr()
# combiny-features
# MMSE+MOCA
# BOSTON+MINT
# CRAFTSTORY+LOGICALMEM

paired_features = [(MMSE, MOCA),
                   (BOSTON, MINT),
                   (CRAFTSTORY, LOGICALMEM)]

########################################################################

# calculate standard derivations of each data column
# importantly, we only filter for CONTROL SAMPLES
stds = {}
means = {}
for i in all_features:
    # we filter for < 80 as special keys
    tmp_data = data[CTL_key & data[i]<80][i]
    stds[i] = tmp_data[tmp_data != -4].std()
    means[i] = tmp_data[tmp_data != -4].mean()

# now, we normalize all data gainst these samples
data_normed = []
# for each pair of features, we use the second element
# in the pair to supplement the first element in the pair
# we also perform the actual normalization
for a,b in paired_features:
    normed = (data[a]-means[a])/stds[a]
    normed.loc[data[a] == -4] = ((data[b]-means[b])/stds[b]).loc[data[a] == -4]
    data_normed.append(normed)

# we now have the normed data here. Let's now create a dataframe about it
df = pd.DataFrame(index=data.index)
# type: ignore
for i, feature in enumerate(data_normed):
    df[f"feature_{i}"] = feature

########################################################################

# simple classification
x_train, x_test, y_train, y_test = train_test_split(df, output_key, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# go!
clsf = GradientBoostingClassifier()
clsf = clsf.fit(x_train, y_train)
# report
preds = clsf.predict(x_val)
print(classification_report(y_val, preds))

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

# get the neuralpsych variables
with open("./neuralpsych", 'r') as df:
    lines = df.readlines()
    lines = [i.strip() for i in lines]

NEURALPSYCH = lines

########################################################################

# for i in range(len(NEURALPSYCH)):
    # for j in range(i+1, len(NEURALPSYCH)):
i = 0
j = 1
var_a = NEURALPSYCH[i]
var_b = NEURALPSYCH[j]

# get data for both columns
var_a_data = data[var_a]
var_b_data = data[var_b]

# checking columns for which both exists
both = (var_a_data != -4) & (var_b_data != -4)

# and getting those columns
both_data = data[[var_a, var_b]][both]
both_data



from os import R_OK
from matplotlib.cbook import _reshape_2D
import pandas

import pandas as pd
from pandas.arrays import BooleanArray
# type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
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

import difflib

import random

from matplotlib import pyplot as plt

# pandas tqdm
tqdm.pandas()

# Read the raw dataset
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

table = []
missing_both = []
for i in tqdm(range(len(NEURALPSYCH))):
    for j in range(i+1, len(NEURALPSYCH)):
        var_a = NEURALPSYCH[i]
        var_b = NEURALPSYCH[j]

        # get data for both columns
        var_a_data = data[var_a]
        var_b_data = data[var_b]

        # checking columns for which both exists
        both = (var_a_data != -4) & (var_b_data != -4)

        # and getting those columns
        both_data = data[[var_a, var_b]][both]

        if len(both_data) < 5:
            missing_both.append((var_a, var_b))
            continue

        try:
            # we reshape because we are using "one feature" to predict another
            reg = LinearRegression().fit(both_data[var_a].values.reshape(-1,1), both_data[var_b])
            table.append((var_a, var_b, reg.score(both_data[var_a].values.reshape(-1,1), both_data[var_b]), len(both_data)))
        except ValueError:
            print("ERROR!", var_a, var_b)

linear_corr = pd.DataFrame(table)
linear_corr.columns = ["Feature1", "Feature2", "R2", "N"]
linear_corr = linear_corr.sort_values(by="R2", ascending=False)
linear_corr.to_csv("./correlate_pairs.csv", index=False)

########################################################################

# we use how similar the names are as a heuristic of how close the features
# are in terms of what they are measuring. TODO probably better idea somewhere else
# but hey, #it_works.

# we don't compare the letters "NACC" as that information is often put in various places
linear_corr["FeatTextSim"] = linear_corr.apply((lambda x: difflib.SequenceMatcher(None,
                                                                                  x.Feature1.replace("NACC", ""),
                                                                                  x.Feature2.replace("NACC", "")).ratio()),
                                               axis=1)
# we want features with max R2 and min similarity; so 1- similarity ration
# however, we weight R^2 being high slightly higher than the features being
# dissimilar. Otherwise, we will get results with absolutely different features
# and disasterous R^2 being scored high
R2_WEIGHT = 0.55
SIM_WEIGHT = 0.45

linear_corr["Score"] = R2_WEIGHT*linear_corr["R2"] + SIM_WEIGHT*(1-linear_corr["FeatTextSim"])
linear_corr = linear_corr.sort_values(by="Score", ascending=False)
linear_corr.to_csv("./correlate_pairs_scored.csv", index=False)

linear_corr


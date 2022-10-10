# exploration anew!

# type: ignore
from os import times_result
import numpy as np
import pandas as pd
from pandas.core.common import random_state
# type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# classification report
from sklearn.metrics import classification_report
# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# type: ignore
# stats
from scipy.stats import kstest, pearsonr, f_oneway, chi2_contingency
from scipy import sparse
from tqdm import tqdm
# sns
import seaborn as sns
# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split, validation_curve

import random

from matplotlib import pyplot as plt

# Read the raw data
data = pd.read_csv("./investigator_nacc57.csv")

#######################################################################

# augment the data

## age group
data["age_group"] = ((data["NACCAGE"]//10)*10).astype(int)

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

# Timeseries data
timeseries_count = {j:len(data.loc[j]) for j in tqdm([i[0] for i in data.index])}
timeseries_count = pd.Series(timeseries_count)
timeseries_count.value_counts()

########################################################################

RACE = data.NACCNIHR 

# groupby race
data.groupby("Participant ID").first().NACCNIHR.value_counts()

data.OTRAILA.value_counts()

#######################################################################

# for each feature, calculate their chi2 contingency with the cognitive status data
# drap non-numeric clomuns and nansj
cleaned_data = data.drop(cleaned_data.select_dtypes(exclude='number'), axis=1)
cleaned_data = cleaned_data.dropna()

# redo keys with drops
DIAGNOSES = cleaned_data.NACCETPR
COG_STATUS = cleaned_data.NACCUDSD

def try_chi_squared(X, y):
    try: 
        return chi2_contingency(pd.crosstab(X, y))[1]
    except ValueError:
        return -100

cog_status_corell = cleaned_data.apply(lambda x: try_chi_squared(x, COG_STATUS), axis=0)
cog_status_corell = cog_status_corell.sort_values()

# we drop some columns from this info
cleaned_data_drop = cleaned_data.drop(columns=["NACCETPR", "NACCUDSD", "NACCADC",
                                               "NACCDAYS", "DECAGE", "NACCYOD",
                                               "NACCFDYS", "PDYR", "HATTYEAR",
                                               "BEAGE", "PDOTHRYR", "TBIYEAR",
                                               "NACCDSYR", "MOAGE", "NACCINT",
                                               "NACCINT", "NACCDAGE", "PARKAGE",
                                               "COGFLAGO", "ALSAGE", "NACCIDEM",
                                               "NACCMCII", "BEVHAGO", "BEREMAGO",
                                               "NACCLBDE", "QUITSMOK", "INBIRYR",
                                               "NACCALZD", "NACCPPA", "NACCCOGF",
                                               "COURSE", "FRSTCHG", "NACCBVFT",
                                               "NACCLBDS", "PPAPH", "VASC",
                                               "COGMODE", "NACCMCIA", "NACCMCIV",
                                               "POSSAD", "FTD", "ALCDEM", "DEMUN",
                                               "NACCMOD", "NACCBEHF", "NACCMCIL",
                                               "NACCMCIE", "NACCALZP", "PROBAD",
                                               "NACCMOTF", "IMPNOMCI", "VASCPS",
                                               "MOMODE", "INHISPOR", "INRASEC",
                                               "BEMODE", "NACCTMCI", "INRATER",
                                               "COGMEM", "NACCSTYR", "NACCDSMO",
                                               "COGJUDG", "NACCTIYR", "NORMCOG",
                                               "NACCDSDY", "NACCNORM", "INKNOWN",
                                               "NACCAVST", "NACCNVST", "INBIRMO",
                                               "LOGIYR", "INEDUC"])
# other ones: BEVHAGO, BEREMAGO

## !!!PEOPLE WHO LATER GET DEMENITA: NACCIDEM, NACCMCII

#######################################################################

# experiment #1:

# feature selection
kbest = SelectKBest(chi2, k=20).fit(abs(cleaned_data_drop), COG_STATUS) # for non-negativity

# get the input features
in_features = kbest.get_feature_names_out()

# split train test
x_train, x_test, y_train, y_test = train_test_split(cleaned_data, COG_STATUS, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# and now, we predict
clsf = RandomForestClassifier(random_state=42)
clsf = clsf.fit(x_train[in_features], y_train)

# and print a report
print(classification_report(clsf.predict(x_val[in_features]), y_val))

#######################################################################

# experiment #2:

# feature selection
# abs() for non-negativity
kbest_overall = SelectKBest(chi2, k=20).fit(abs(cleaned_data_drop), COG_STATUS) 

drop_dementia = COG_STATUS != 4
kbest_dd = SelectKBest(chi2, k=20).fit(abs(cleaned_data_drop)[drop_dementia], COG_STATUS[drop_dementia])
drop_control = COG_STATUS != 1
kbest_mci_inp = SelectKBest(chi2, k=20).fit(abs(cleaned_data_drop)[drop_dementia&drop_control], COG_STATUS[drop_dementia&drop_control])

# get the input features
in_features_overall = kbest_overall.get_feature_names_out()
in_features_dd = kbest_dd.get_feature_names_out()
in_features_mci_inp = kbest_mci_inp.get_feature_names_out()

in_features = list(set((np.concatenate([in_features_overall, in_features_dd, in_features_mci_inp]))))

# split train test
x_train, x_test, y_train, y_test = train_test_split(cleaned_data, COG_STATUS, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# and now, we predict
clsf = RandomForestClassifier(random_state=42)
clsf = clsf.fit(x_train[in_features], y_train)

# and print a report


max(x_val.age_group)

for age_group in range(0, 110, 10):
    if len(y_val[x_val.age_group == age_group]) > 0:
        print(f"for {age_group}")
        print(classification_report(y_val[x_val.age_group ==age_group], clsf.predict(x_val[x_val.age_group ==age_group][in_features]), labels=[1,2,3,4]))
    


data.NACCDIED.value_counts()

# exploration anew!

# type: ignore
from os import times_result
import pandas as pd
# type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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

import random

from matplotlib import pyplot as plt

# pandas tqdm
tqdm.pandas()

# Read the raw data
data = pd.read_csv("./investigator_nacc57.csv")

########################################################################

# Variable Names and Explanations
# hispanic, living situation, derived race, years of education
democraphics = ["HISPANIC", "NACCLIVS", "NACCNIHR", "EDUC"]

# Let's select for other participant info
# sex, current age, level of independence
info = ["SEX", "NACCAGE", "INDEPEND"]

# Hereditary history
# mother with cogimp, father with coginp, family of AD mutation, family of FTLD mutation
hereditary_history = ["NACCMOM", "NACCDAD", "NACCFADM", "NACCFFTD"]

# Drug use
# hypertentive medication, hypertentative combo, ACE inhibiter, antiandrenic, beta-andretic agent, calcium channel agent, diuretic, vasodilator, angiotensin II inhibitor, lipid lowering medication, NSAIDs, anticoagulant, antidepressent, antipsycotics, sedative, [skip NACCADMD NACCPDMD for they are active treatment for altimers and parkinsons], estrogen hormone theapy, estrogen + progestin therapy, diabetes
drug_use = ["NACCAHTN", "NACCHTNC", "NACCACEI", "NACCAAAS", "NACCBETA", "NACCCCBS", "NACCDIUR", "NACCVASD", "NACCANGI", "NACCLIPL", "NACCNSD", "NACCAC", "NACCADEP", "NACCAPSY", "NACCAANX", "NACCEMD", "NACCEPMD", "NACCDBMD"]

# Behavioral History
# last 30 day smoke, smoke more than 100, smoking age, packs smoked per day, alcahol consumptio, alcahol frequency last 30 days
behaviorial = ["TOBAC30", "TOBAC100", "SMOKYRS", "PACKSPER", "ALCOCCAS", "ALCFREQ"]

# Cardiovascular
# heart attack, more than 1 heart attack, AFIB, angioplasty/stent, cardiac bypass, pacemaker/defib, pacemaker, congestic heart failure, angina, heart valve repair, other disease
heart = ["CVHATT", "HATTMULT", "CVAFIB", "CVANGIO", "CVBYPASS", "CVPACDEF", "CVPACE", "CVCHF", "CVANGINA", "CVHVALVE", "CVOTHR"]

# Brain/Neurological
# stroke, TIA, TBI, Brain trauma with unconciousness, brain trauma with >5 min unconciounsness, brain trauma with unconciousness, TBI with no loss of unconvciousness, seizures
brain = ["CBSTROKE", "CBTIA", "TBI", "TBIBRIEF", "TBIEXTEN", "TRAUMEXT", "SEIZURES"]

# Other Medical
# Diabetes, Hypertension, Hypercholesterolemia, Vitamin b12 deficiency, Thyroid disease, Arthritis, Urinary Incontinence, Incontinence — bowel, Sleep apnea, REM sleep behavior disorder 
medical_misc = ["DIABETES", "HYPERTEN", "HYPERCHO", "B12DEF", "THYROID", "ARTHRIT", "INCONTU", "INCONTF"]

# Mental Health
# Alcohol abuse, bipolar disorder, Schizophrenia, Active depression in the last two years, Depression episodes more than two years ago, Anxiety, OCD, Spectrum Disorder
mental = ["ALCOHOL", "PTSD", "BIPOLAR", "SCHIZ", "DEP2YRS", "DEPOTHR", "ANXIETY", "OCD", "NPSYDEV"]

# Habilititation
# In the past four weeks, did the subject have any difficulty or need help with:  *
# bills, taxes, shopping, games, stove, meal, events, paying attention, remembering dates, trave l
habil = ["BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE", "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"]

# Neuralpsycological Tests
# generic list of neuralpsycological test
with open("./neuralpsych", 'r') as df:
    neuralpsych = df.read().strip().split('\n')

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

# cosntruct a "diagnosis-in-nyears" column
age_date_data = data[["NACCUDSD", "NACCAGE"]]

def find_data(grp):
    """find the age til dementia

    grp: DataFrameGroupBy 
    """

    # get dementia indicies
    # dementia_indicies = grp[(grp.NACCUDSD==3) | (grp.NACCUDSD==4)]
    dementia_indicies = grp[(grp.NACCUDSD==4)]

    # if length is 0 "the person never had dementia"
    if len(dementia_indicies) == 0:
        # make the dementia age 1000 ("forever in the future")
        initial_dementia_age = 1000
    else:
        # get the first age 
        initial_dementia_age = dementia_indicies.iloc[0].NACCAGE

    # get the number of years to the age of dementia
    age_til_dementia = (initial_dementia_age - grp.NACCAGE).apply(lambda x:0 if x<0 else x)

    return age_til_dementia

initial_dementia_age = (age_date_data
                        .groupby(level=0,axis=0)
                        .progress_apply(find_data))
initial_dementia_age.index = data.index
data.loc[:,"age_til_dementia"] = initial_dementia_age

# measure age in dementia
(data.age_til_dementia > 20).value_counts() - 93135

# if this value > 100, it is actually a control sample
# i.e. patient never got dementia
# so we will use this as the contro label
data.loc[:, "timeseries_label"] = (~(data.age_til_dementia > 100))

########################################################################

# Primary Diagnoses Classes

# MCI
MCI_key = data.NACCUDSD == 3
# HOLY HELL I FONUD IT
DIAGNOSES = data.NACCETPR
# drop all missing diags
data = data[DIAGNOSES != 99]
data = data[DIAGNOSES != 30]
# HOLY HELL I FONUD IT
DIAGNOSES = data.NACCETPR
# DIAGNOSES

# data.loc[DIAGNOSES == 1, data.NACCUDSD==3]

###################################################################

# select for dementia that is...
MORE_THAN = -0.1 # years away, but 
LESS_THAN = 0.1 # years away

# for each feature, calculate their chi2 contingency with the cognitive status data
# drap non-numeric clomuns and nansj
cleaned_data = data.drop(data.select_dtypes(exclude=['number', 'bool']), axis=1)
cleaned_data = cleaned_data.dropna()

# get the control samples
control_samples = cleaned_data[~cleaned_data.timeseries_label]

# get the alternate samples
selector = ((cleaned_data.age_til_dementia > MORE_THAN) &
            (cleaned_data.age_til_dementia < LESS_THAN))
cleaned_data = cleaned_data[selector]

# randomly mix in control samples of the same size
cleaned_data = pd.concat([cleaned_data,
                          control_samples.sample(frac=1,
                                                 random_state=42).iloc[:len(cleaned_data)]])

# shuffle again
cleaned_data = cleaned_data.sample(frac=1, random_state=42)

# and get labels
cleaned_labels = cleaned_data.timeseries_label

# we drop some columns from this info except for neuropsych
cleaned_data_drop = cleaned_data[neuralpsych]

# other ones: BEVHAGO, BEREMAGO
X = cleaned_data_drop
y = cleaned_labels

feature_selecter = SelectKBest(k=10).fit(X, y)
(feature_selecter.pvalues_<0.01).sum()

# test tree utilities and reporting
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# get the input features
in_features = feature_selecter.get_feature_names_out() 

# get in/out data
in_data = cleaned_data_drop[in_features]
out_data = cleaned_labels

# split train test
x_train, x_test, y_train, y_test = train_test_split(in_data, out_data, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# fit!
classifier = KNeighborsClassifier()
classifier = classifier.fit(x_train, y_train)

# predict!
preds = classifier.predict(x_val)

import pickle
with open("./maskedata.bin", "wb") as df:
    pickle.dump({"in": in_data, "out": out_data, "features": in_features }, df)



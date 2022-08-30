# type: ignore
import pandas as pd
# type: ignore
from sklearn.tree import DecisionTreeClassifier
# type: ignore
# stats
from scipy.stats import kstest, pearsonr, f_oneway, chi2_contingency
from scipy import sparse
from tqdm import tqdm

import random

# Read the raw data
data = pd.read_csv("./investigator_nacc57.csv")

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

# and all of it that we are interested in, we are just adding them all up here
interest = democraphics + hereditary_history + drug_use + behaviorial + heart + brain + medical_misc + mental + habil

########################################################################

# Get a list of participants
participants = data["NACCID"]

# Drop the parcticipants 
data = data.drop(columns="NACCID")

# Make it a multiindex by combining the experiment ID with the participant
# so we can index by participant as first pass
index_participant_correlated = list(zip(participants, pd.RangeIndex(0, len(data))))
index_multi = pd.MultiIndex.from_tuples(index_participant_correlated, names=["Participant ID", "Entry ID"])
data.index = index_multi

# Choorelation target, which is chimerical
naccalzd = data["NACCALZD"]

# For each feature of interest, we perform a
# one-way chi-square test for choorelation. Note about the
# P-values here: the null hypothesis is that the
# two groups are INDEPENDENT, so a small p-value is desired
# here.
# 
# Therefore, we hope that groups 1, 8, and 0 are significantly
# different in terms of mean in that variable.
#
# We perform chi-square here because most of the input data can be broken
# down into categories to analyze (they are (mostly) not continuous)
# if they are continuous, we will integer quantize it
#
# Furthermore, we create two results with and without ignoring
# naccalzd = 8 because impaired vs. AD is a /very/ different problem
# than normal limits vs. AD

# tabulate results
results = {}

# calculate all results
for feature in tqdm(interest):
    # index the data for the features
    f_index = data[naccalzd != 8][feature]
    # split results into groups
    crosstab = pd.crosstab(naccalzd[naccalzd != 8], round(data[naccalzd != 8][feature]))

    # calculate
    imp_vs_ad = chi2_contingency(crosstab)[1]

    # index the data for the features
    f_index = data[feature]
    # split results into groups
    crosstab = pd.crosstab(naccalzd, round(data[feature]))

    # calculate
    ctrl_vs_ad = chi2_contingency(crosstab)[1]

    # submit!
    results[feature] = {"iva": imp_vs_ad,
                        "nva": ctrl_vs_ad}

# generate a dataframe
results_df = pd.DataFrame(results).transpose()
results_df = results_df.sort_values("iva")

# save results
results_df.to_csv("correlate.csv")

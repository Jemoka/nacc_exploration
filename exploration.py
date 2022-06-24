# type: ignore
import pandas as pd
# type: ignore
from sklearn.tree import DecisionTreeClassifier

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

########################################################################

# Get a list of participants
participants = data["NACCID"]

# Drop the parcticipants 
data = data.drop(columns="NACCID")

# Make it a multiindex
index_participant_correlated = list(zip(participants, pd.RangeIndex(0, len(data))))
index_multi = pd.MultiIndex.from_tuples(index_participant_correlated, names=["Participant ID", "Entry ID"])
data.index = index_multi
data.loc[2]


# The column we are predicting on should be NACCALZD, presumptive
# etiologic diagnosis of the cognitive disorder of Alzheimer’s disease
# ("NACC" "Alzhimer" "Diagnoses" NACCALZD)
naccalzd = data["NACCALZD"]

# 0 - No, 1 - Yes, 8 - N/A (no cog. impairment)
# Let's replace 8 with 2 
naccalzd = naccalzd.replace(8, 2)
# So now
# 0 - No, 1 - Yes, 2 - N/A (no cog. impairment)


#
data

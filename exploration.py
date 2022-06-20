# type: ignore
import pandas as pd
# type: ignore
from sklearn.tree import DecisionTreeClassifier

# Read the raw data
data = pd.read_csv("./investigator_nacc57.csv")

# Get a list of participants
participants = data["NACCID"]

# The column we are predicting on should be NACCALZD, presumptive
# etiologic diagnosis of the cognitive disorder of Alzheimer’s disease
# ("NACC" "Alzhimer" "Diagnoses" NACCALZD)
naccalzd = data["NACCALZD"]

# 0 - No, 1 - Yes, 8 - N/A (no cog. impairment)
# Let's replace 8 with 2 
naccalzd = naccalzd.replace(8, 2)
# So now
# 0 - No, 1 - Yes, 2 - N/A (no cog. impairment)

# Let's select for participant demographics
# hispanic, living situation, derived race, years of education
participant_democraphics = data[["HISPANIC", "NACCLIVS", "NACCNIHR", "EDUC"]]

# Let's select for other participant info
# sex, current age, level of independence
participant_info = data[["SEX", "NACCAGE", "INDEPEND"]]

# Hereditary history
# mother with cogimp, father with coginp, family of AD mutation, family of FTLD mutation
hereditary_history = data[["NACCMOM", "NACCDAD", "NACCFADM", "NACCFFTD"]]

# Drug use
# hypertentive medication, hypertentative combo, ACE inhibiter, antiandrenic, beta-andretic agent, calcium channel agent, diuretic, vasodilator, angiotensin II inhibitor, lipid lowering medication, NSAIDs, anticoagulant, antidepressent, antipsycotics, sedative, [skip NACCADMD NACCPDMD for they are active treatment for altimers and parkinsons], estrogen hormone theapy, estrogen + progestin therapy, diabetes
drug_use = data[["NACCAHTN", "NACCHTNC", "NACCACEI", "NACCAAAS", "NACCBETA", "NACCCCBS", "NACCDIUR", "NACCVASD", "NACCANGI", "NACCLIPL", "NACCNSD", "NACCAC", "NACCADEP", "NACCAPSY", "NACCAANX", "NACCEMD", "NACCEPMD", "NACCDBMD"]]

# Behavioral History
# last 30 day smoke, smoke more than 100, smoking age, packs smoked per day, alcahol consumptio, alcahol frequency last 30 days
behaviorial = data[["TOBAC30", "TOBAC100", "SMOKYRS", "PACKSPER", "ALCOCCAS", "ALCFREQ"]]

# Cardiovascular
# heart attack, more than 1 heart attack, AFIB, angioplasty/stent, cardiac bypass, pacemaker/defib, pacemaker, congestic heart failure, angina, heart valve repair, other disease
heart = data[["CVHATT", "HATTMULT", "CVAFIB", "CVANGIO", "CVBYPASS", "CVPACDEF", "CVPACE", "CVCHF", "CVANGINA", "CVHVALVE", "CVOTHR"]]

# Brain/Neurological
# stroke, TIA, TBI, Brain trauma with unconciousness, brain trauma with >5 min unconciounsness, brain trauma with unconciousness, TBI with no loss of unconvciousness, seizures
brain = data[[]]





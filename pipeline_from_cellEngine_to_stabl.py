
import pandas as pd 
import numpy as np
from pathlib import Path
import sys
import csv

changetimepoint = False #this will be useful later. We use that bc we might need to delete some columns, at the end, that are created with a stim and timepoint not compatible. In that case we will go across timepoint and delete inappropriate columns. In case we have changed the name of the timepoints we need to be sure of that/

def detect_separator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(1024)
        try:
            return csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            return ','  # fallback par défaut

# PATH FOR USER
freq_path = Path("./frequency.csv")
func_path = Path("./functional.csv")
freq_sep = detect_separator(freq_path)
func_sep = detect_separator(func_path)
excel_file_path = "./penalization.xlsx"  #FOR PENALIZATION

# LOAD
frequency = pd.read_csv(freq_path, sep=freq_sep)
functional = pd.read_csv(func_path, sep=func_sep)
 
# MANDATORY COLUMNS
required_columns_funct = ["sampleID", "population", "reagent", "median"]
required_columns_freq = ["sampleID", "population", "reagent", "frequency"]
# Missing columns
missing_columns_funct = [col for col in required_columns_funct if col not in functional.columns]
missing_columns_freq = [col for col in required_columns_freq if col not in frequency.columns]

if missing_columns_funct:
    print("\nThese columns are missing in your functional file:", missing_columns_funct)
    print("\nYour current functional columns are :", functional.columns.tolist())
if missing_columns_funct:
    print("\nThese columns are missing in your frequency file:", missing_columns_freq)
    print("\nYour current frequency columns are", frequency.columns.tolist())

print("\nthe following columns will be deleted if they still exists : fcsFileId, populationId, filename, uniquePopulationName, parentPopulation, parentPopulationId, channel")

# No time column = 0 time point
if "time" not in functional.columns:
    functional["time"] = 0
    print("\nColumns 'time' was not found : it was created with value 0.")

######## MEDIAN AND FREQUENCY ARE MADE OF FLOATS ########
# MEDIAN
if not pd.api.types.is_float_dtype(functional['median']):
    print(f"\n 'median' is not of float type. Please ensure the values are numeric.")

    # If str
    if functional['median'].apply(lambda x: isinstance(x, str)).any():
        print('\n Detected some string values in median')
        functional['median'] = functional['median'].fillna(0) #just in case but might need to change that.
        functional['median'] = functional['median'].astype(str)

        # If there is some ',' we replace them by '.'
        functional['median'] = functional['median'].str.replace(',', '.', regex=False)
        try:
            functional['median'] = pd.to_numeric(functional['median'], errors='raise')
            print(f"Conversion done (',' had to be replaced by '.')")
        except Exception as e:
            raise ValueError(f"\n FAIL: Unable to convert 'median' column to float.\nYou need to check this error message: {e}")
    else:
        try:
            functional['median'] = pd.to_numeric(functional['median'], errors='raise')
            print(f"Median converted")
        except Exception as e:
            raise ValueError(f"\n Unable to convert 'median' column to float.\nYou need to check this error message: {e}")
else:
    print("\n median is made of float.")

#FREQUENCY

if not pd.api.types.is_float_dtype(frequency['frequency']):
    print(f"\n 'frequency' is not of float type. Please ensure the values are numeric.")

    # If str
    if frequency['frequency'].apply(lambda x: isinstance(x, str)).any():
        print('\n Detected some string values in frequency')
        frequency['frequency'] = frequency['frequency'].fillna(0) #just in case but might need to change that.
        frequency['frequency'] = frequency['frequency'].astype(str)

        # If there is some ',' we replace them by '.'
        frequency['frequency'] = frequency['frequency'].str.replace(',', '.', regex=False)
        try:
            frequency['frequency'] = pd.to_numeric(frequency['frequency'], errors='raise')
            print(f"Conversion done (',' had to be replaced by '.')")
        except Exception as e:
            raise ValueError(f"\n FAIL: Unable to convert 'frequency' column to float.\nYou need to check this error message: {e}")
    else:
        try:
            frequency['frequency'] = pd.to_numeric(frequency['frequency'], errors='raise')
            print(f"frequency converted")
        except Exception as e:
            raise ValueError(f"\n Unable to convert 'frequency' column to float.\nYou need to check this error message: {e}")
else:
    print("\n frequency is made of float.")

############# TIME ISSUE 1 ###############

    # checking 'time'
if not pd.api.types.is_integer_dtype(functional['time']):
    print(f"\nTime is not made of int values. BE SURE TIME IS THE SAME FOR FREQUENCY AND FUNCTIONAL")
    print("\nHere are the time value detected :")
    print(functional['time'].unique())

    # asking user what to replace with 
    changetimepoint = True #cf. explanation at the beginning.
    replacements = {}
    for val in functional['time'].unique():
        try:
            int(val)
        except:
            new_val = input(f"Quel entier doit remplacer '{val}' ? ")
            replacements[val] = int(new_val)
    functional['time'] = functional['time'].replace(replacements).astype(int)
    frequency['time'] = frequency['time'].replace(replacements).astype(int)
    print(f"Replacement done")


############# TIME ISSUE 1 END ###############


# No stim columns = 'unstim' columns
if "stimulation" not in functional.columns:
    functional["stimulation"] = "Unstim"
    print("\nColumns stim was not found :it was created with value 'Unstim'.")


##### GROUP/OUTCOME ISSUE ##### we handle missing group column, binary that would not be made of 0 and 1, and a group that would not be named 'group'.
# Step 1: Check if the outcome column exists.

groupExistsFlag = True

if "group" not in functional.columns:
    response = input("\nNo 'group' column found. Does your outcome column exist under a different name? (yes/no): ").strip().lower()
    if response == "yes":
        other_name = input("\nPlease enter the name of the outcome column: ").strip()
        if other_name in functional.columns:
            functional = functional.rename(columns={other_name: "group"})
            print(f"Column '{other_name}' has been renamed to 'group'.")
        else:
            print(f"Column '{other_name}' does not exist in your data. Exiting.")
            sys.exit()
    else:
        print("\nNo outcome column provided. Creating an artificial 'group' column with all 0s.")
        functional["group"] = 0
        groupExistsFlag = False

# Step 2: Ask if the outcome is binary.
binary_response = input("\nIs your outcome binary? (yes/no) [default: yes]: ").strip().lower() or "yes"
if binary_response == "yes":
    # Outcome is supposed to be binary. Verify if group column contains only 0 and 1.
    unique_groups = functional["group"].unique()
    if not set(unique_groups).issubset({0, 1}):
        functional["group"] = functional["group"].astype(str)
        frequency["group"] = frequency["group"].astype(str)
        print("\nCurrent 'group' values are:", unique_groups)
        control = input("\nPlease indicate which value should be replaced by 0 (control group): ").strip()
        case = input("\nPlease indicate which value should be replaced by 1 (case group): ").strip()
        functional["group"] = functional["group"].map({control: 0, case: 1}).fillna(functional["group"])
        frequency["group"] = frequency["group"].map({control: 0, case: 1}).fillna(frequency["group"])
        print("\nNew 'group' values (should be 0 and 1):", functional["group"].unique(), frequency["group"].unique())
    else:
        print("\nThe 'group' column is already binary (0 and 1).")
else:
    # Outcome is not binary. We need to convert it to binary using a threshold.
    print("\nCurrent outcome values in 'group':", functional["group"].unique())
    threshold_input = input("Please specify the threshold: values below this threshold will be set to 0, and values equal or above will be set to 1: ").strip()
    try:
        threshold = float(threshold_input)
    except ValueError:
        print("Invalid threshold provided. Exiting.")
        sys.exit()
    functional["group"] = functional["group"].apply(lambda x: 0 if x < threshold else 1)
    print("Outcome has been converted to binary. New 'group' values:", functional["group"].unique())
print(f"this is group for functional : {functional['group'].unique()}")

##### GROUP/OUTCOME ISSUE END #####


# Processing functional/frequency values : largely inspired by the R code that can be found on Notion.
# Suppressing useless columns according to the manual gating in Notion
cols_to_drop_functional = ['fcsFileId', 'populationId', 'filename', 
                           'uniquePopulationName', 'parentPopulation', 
                           'parentPopulationId', 'channel']
existing_cols_functional = [col for col in cols_to_drop_functional if col in functional.columns]
functional = functional.drop(columns=existing_cols_functional)

# Suppressing useless columns according to the manual gating in Notion
cols_to_drop_frequency = ['fcsFileId', 'populationId', 'filename', 
                          'uniquePopulationName', 'parentPopulation', 
                          'parentPopulationId', 'channel', 'percentOfId', 
                          'percentOf', 'reagent']
existing_cols_frequency = [col for col in cols_to_drop_frequency if col in frequency.columns]
frequency = frequency.drop(columns=existing_cols_frequency)

# dropping any other column that is not stim, pop, time, group, reagent, sampleID, median, feature, frequency
 
columns = functional.columns.tolist()
keepcolumns = ["stimulation", "population", "time", "group", "reagent", "sampleID", "median", "feature", "frequency"]
functional = functional.drop(columns=[col for col in columns if col not in keepcolumns])
print(f"\nfunctional now have these columns : {functional.columns}")

#just copying

data = functional.copy()
finaldata = pd.DataFrame() 
# timepoints and stims
times = data['time'].drop_duplicates()
stims = data['stimulation'].drop_duplicates()
print(f"\nWe have the following stims: {stims.tolist()}")

# Changing median column
data['median'] = np.arcsinh(data['median']/5)
data.rename(columns={'median': 'feature'}, inplace=True)

#large format pivot : now one stim per column
datat_wide = data.pivot(
    index=["sampleID", "time", "population", "reagent", "group"],
    columns="stimulation",
    values="feature"
).reset_index()
print("Columns after pivot:", datat_wide.columns.tolist())
print(f"this is group for functional4 : {functional['group'].unique()}")
# Just in case, to ensure the pivot is okay : 
for stim in stims:
    if stim not in datat_wide.columns:
        response = input(f"We have found that the stimulation '{stim}' is missing in the pivoted data. Do you want to continue? (yes/no): ")
        if response.strip().lower() != 'yes':
            print("Cancelled due to missing stimulation columns.")
            sys.exit()
# Stim ratio
for stim in stims:
    if stim != 'Unstim':
        datat_wide[stim] = datat_wide[stim] - datat_wide['Unstim']

print(datat_wide.head())

# back to long format : melt is the opposite of pivot, we put stims back in a single column
datafin = datat_wide.melt(
    id_vars=["sampleID", "time", "population", "reagent", "group"],  
    value_vars=stims,        
    value_name="feature"
)

# Functional and frequency 
finaldata = pd.concat([finaldata, datafin], ignore_index=True)
print(f"\nthis is group for frequency : {frequency['group'].unique()}")

# We add the frequency column in the good shape.
frequency = frequency.rename(columns={'frequency': 'feature'})
frequency['reagent'] = 'frequency'
print(frequency.head())
finaldata = pd.concat([finaldata, frequency], ignore_index=True)

#does the user want some frequency ? 

if 'reagent' in finaldata.columns:
    condition = (finaldata['reagent'].str.lower() == 'frequency') & (finaldata['stimulation'] != 'Unstim')
    if condition.any():
        response = input("Some rows in 'finaldata' have 'reagent' equal to 'frequency' while having a stimulation other than 'Unstim'. Do you want to keep them? (yes/no): ").strip().lower()
        if response == "no":
            finaldata = finaldata[~condition]  
            print("Rows with 'reagent' equal to 'frequency' and stimulation not equal to 'Unstim' have been removed.")

finaldata = finaldata.reset_index(drop=True)


# UX CHECKPOINT 2: checking dataframe
print(f"\nthis is group for finaldata : {finaldata['group'].unique()}")

print("\nYour preprocess file looks like :")
print(finaldata.head())
print("\nIt has the following shape", finaldata.shape)

# Sauvegarde du fichier prétraité
out_path = Path("Preprocess")
out_path.mkdir(parents=True, exist_ok=True) #creating the file Preprocess if it does not exist. if already exist, replace previous one. 
csv_file_path = out_path / "preprocessed.csv"
print(f"\nthis is group for finaldata : {finaldata['group'].unique()}")

finaldata.to_csv(csv_file_path, index=False)

data = finaldata #now working withe preprocess file.

####### FROM HERE WE HAVE A 'PREPROCESSED' FILE ##############
####### FROM HERE WE HAVE A 'PREPROCESSED' FILE ##############
####### FROM HERE WE HAVE A 'PREPROCESSED' FILE ##############
####### FROM HERE WE HAVE A 'PREPROCESSED' FILE ##############
####### FROM HERE WE HAVE A 'PREPROCESSED' FILE ##############


penalization_dfs = pd.read_excel(excel_file_path, sheet_name=None)  
penalization_list = []

# Récupérer les populations uniques présentes dans le DataFrame "functional"
data_populations = set(functional['population'].drop_duplicates()) #drop duplicates and set operate the same way, we cuould have used only 'set' actually

print("\nComparing cell population between Penal Matrix and Preprocess file")
print(f"\n FYI Stimulations that are present in your data but not in the penalization matrix will be kept.")

for stim, df in penalization_dfs.items():
    # Renaming "Population"
    df = df.rename(columns={df.columns[0]: "population"})
    # Melt allows us to put the feature in the same column. One row is now : Population - Feature - Penalty value
    melted = df.melt(id_vars="population", var_name="feature", value_name="penalty")
    melted["stimulation"] = stim  # Ajout de la colonne stimulation
    penalization_list.append(melted)
    # Comparing population : just checkinf that everything is okay.
    matrix_populations = set(df["population"].unique())
    missing_in_data = matrix_populations - data_populations
    missing_in_matrix = data_populations - matrix_populations
    if missing_in_data:
        print(f"\n \nFYI, for stimulation '{stim}', the following populations are in the penalization matrix but not in the functional: {missing_in_data}")  
    if missing_in_matrix:
        print(f"\n \nFYI, for stimulation '{stim}', the following populations are in the functional but not in the penalization matrix: {missing_in_matrix}")
    else:
        print(f"no missing population between penal and data for '{stim}' ")
# Checking if penalization stims match with the stims in the preprocess file.

penalization_df = pd.concat(penalization_list, ignore_index=True)
penal_stims = set(penalization_df["stimulation"].unique())
preprocess_stims = set(stims)
missing_stims = preprocess_stims - penal_stims
if missing_stims:
    print(f"\n \nWarning: The following stimulations are present in your data but not in the penalization matrix: {missing_stims}. They will not be penalized.")
else:
    print("no difference of stims between penal and data")


# 2. Renaming : it is a bit useless now. @DELETE    

features = pd.read_csv("Preprocess/preprocessed.csv")
features = features.rename(columns={
    "reagent": "feature",
    "feature": "value",
})

features['feature'] = features['feature'].apply(
    lambda x: x.split('_')[1] if '_' in x and len(x.split('_')) > 1 else x
)
print(features['feature'].head())
    
# 3. merge + filter : since features and penal both have population, feature and and stim column we can merge them on these column so the filter is easy.
merged = pd.merge(
    features,
    penalization_df,
    on=["population", "feature", "stimulation"],
    how="left"
)
merged["penalty"] = merged["penalty"].fillna(1) #Very important: this means that all stim/pop that are not in the penal are not penalized
merged = merged[merged["penalty"] != 0] #Filtering
 
# 4. back to long format
 
pivoted = merged.pivot_table(
    index=["sampleID", "time", "group"], #these don't move
    columns=["population", "feature", "stimulation"], #All stim and all features now have a column.
    values="value" #this is where we put the feature value
).reset_index()
# creating columns names
pivoted.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col 
    for col in pivoted.columns.values
]
pivoted = pivoted.rename(columns={'time__': 'time', 'sampleID__': 'sampleID', 'group__': 'group'}) #this is to correct the renaming of the column that was made in the two lines above (bc it had a _ instead of space)

################
#TIME ISSUE #In case there are some time value that does not have the same stim than others we ensure that no artificial column would be created for example if you have way more stim for the first TP
################

# What we don't touch
id_vars = {"sampleID", "time", "group"}

# timepoints
unique_times = pivoted["time"].unique()

# # Pour chaque timepoint, filtrer les colonnes dont le suffixe (stim) est autorisé pour ce timepoint
for t in unique_times:
    # we select the appropriate line with a given timepoint
    time_data = pivoted[pivoted["time"] == t]
    print(time_data.head())


    if changetimepoint:
         t = replacements.get(t, t)
    #for this time point we now identify the authorized stimulation
    allowed_stims = set(functional[functional["time"] == t]["stimulation"].unique())
    print(f"For timepoint {t} , allowed stimulations (from functional) are: {allowed_stims}")
    

    #what we're going to check
    cols_to_check = [col for col in time_data.columns if col not in id_vars]
    cols_to_drop = []
    
    # we extract the stim
    for col in cols_to_check:
        parts = col.split('_')
        if len(parts) >= 3:
            # we take the last part because this is where there's the stim and we won't have problem with a potential metal name that would not have been deleted.
            col_stim = parts[-1]
            if col_stim not in allowed_stims:
                cols_to_drop.append(col)
        else:
            print(f"Columns name : '{col}' is not in the good shape so we'll ignore the check-up ")
    
    if cols_to_drop:
        print(f"Timepoint {t}: it is likely there is not as much stim for this time point that for the other")
        time_data = time_data.drop(columns=cols_to_drop)
    else:
        print(f"Timepoint {t}: all stimulation columns are consistent with the preprocessed data.")

########################
# Time stim issue end.
########################

    # Extraire l'outcome
    if groupExistsFlag:
        outcome = time_data['group'] #there is smth to do here bc what about the outcome is in another file because it is not binary.
        outcome.to_csv(f"outcome{t}.csv", index=False, sep=",")
    if groupExistsFlag == False:
        print("You likely have your own outcome in another file.")
        response = input("Do you want to provide your own outcome file? (yes/no): ").strip().lower()

        if response == "yes":
            outcome_path = input("Please provide the path to your outcome CSV file: ").strip()
            try:
                outcome_df = pd.read_csv(outcome_path)
            except Exception as e:
                print(f"Failed to load the outcome file: {e}")
                sys.exit()

            if "sampleID" not in outcome_df.columns:
                print("Your outcome file must contain a 'sampleID' column.")
                sys.exit()

            samples_data = set(time_data["sampleID"])
            samples_outcome = set(outcome_df["sampleID"])

            not_in_outcome = samples_data - samples_outcome
            not_in_data = samples_outcome - samples_data

            if not_in_outcome:
                print(f"Samples present in time_data but not in outcome: {not_in_outcome}")
            if not_in_data:
                print(f"Samples present in outcome but not in time_data: {not_in_data}")

            if not_in_outcome or not_in_data:
                answer = input("Do you want to keep only samples that are present in both files? (yes/no): ").strip().lower()
                if answer == "yes":
                    common_samples = samples_data & samples_outcome
                    time_data = time_data[time_data["sampleID"].isin(common_samples)]
                    outcome_df = outcome_df[outcome_df["sampleID"].isin(common_samples)]
                else:
                    print("Inconsistent samples will not be resolved. Exiting.")
                    sys.exit()

            outcome_df.to_csv(f"outcome{t}.csv", index=False, sep=",")
        else:
            print("No external outcome file provided. Skipping outcome export.")
        # Sauvegarder les données sans la colonne "group" et "time"
        time_data.drop(columns=["group", "time"]).to_csv(f"data{t}.csv", index=False, sep=",")

print("All done")

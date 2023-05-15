# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import gc

import wandb

import functools

from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import KFold

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random

bound=(1,3)

# loading data
class NACCCurrentDataset(Dataset):

    def __init__(self, file_path, feature_path,
              # skipping 2 impaired because of labeling inconsistency
                 target_feature="NACCUDSD", target_indicies=[1,3,4],
                 emph=3, fold=0):
        """The NeuralPsycology Dataset

        Arguments:

        file_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_feature] (str): the name of feature to serve as the target
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [emph] (int): the index to emphasize 
        [fold] (int): the n-th fold to select
        """

        # initialize superclass
        super(NACCCurrentDataset, self).__init__()

        # Read the raw dataset
        self.raw_data = pd.read_csv(file_path)

        # get the fature variables
        with open(feature_path, 'r') as df:
            lines = df.readlines()
            self.features = [i.strip() for i in lines]

        # skip elements whose target is not in the list
        self.raw_data = self.raw_data[self.raw_data[target_feature].isin(target_indicies)] 
        # Get a list of participants
        participants = self.raw_data["NACCID"]

        # Drop the parcticipants 
        self.raw_data = self.raw_data.drop(columns="NACCID")

        # Make it a multiindex by combining the experiment ID with the participant
        # so we can index by participant as first pass
        index_participant_correlated = list(zip(participants, pd.RangeIndex(0, len(self.raw_data))))
        index_multi = pd.MultiIndex.from_tuples(index_participant_correlated, names=["Participant ID", "Entry ID"])
        self.raw_data.index = index_multi

        # k fold
        participants = self.raw_data.index.get_level_values(0)

        kf = KFold(n_splits=10, random_state=7, shuffle=True)

        participant_set = pd.Series(list(set(participants)))
        splits = kf.split(participant_set)
        train_ids, test_ids = list(splits)[fold]

        train_participants = participant_set[train_ids]
        test_participants = participant_set[test_ids]

        # shuffle
        self.raw_data = self.raw_data.sample(frac=1)

        # Calculate the target data
        self.targets = self.raw_data[target_feature]
        self.data = self.raw_data[self.features] 

        # store the traget indicies
        self.__target_indicies = target_indicies

        # get number of features, by hoisting the get function up and getting length
        self._num_features = len(self.features)

        # crop the data for validatino
        self.val_data = self.data.loc[test_participants]
        self.val_targets = self.targets.loc[test_participants]

        self.data = self.data.loc[train_participants]
        self.targets = self.targets.loc[train_participants]

        # basic dataaug
        if emph:
            emph_features = self.targets==emph

            ordering = len(self.data)
            order = random.sample(range(ordering), ordering)

            self.data = pd.concat([self.data, self.data[emph_features]]).iloc[order]
            self.targets = pd.concat([self.targets, self.targets[emph_features]]).iloc[order]

    def __process(self, data, target, index=None):
        # the discussed dataprep
        # if a data entry is <0 or >80, it is "not found"
        # so, we encode those values as 0 in the FEATURE
        # column, and encode another feature of "not-found"ness
        data_found = (data > 80) | (data < 0)
        data[data_found] = 0
        # then, the found-ness becomes a mask
        data_found_mask = (data_found)

        # if it is a sample with no tangible data
        # well give up and get another sample:
        if sum(~data_found_mask) == 0:
            if not index:
                raise ValueError("All-Zero found in validation!")
            indx = random.randint(2,5)
            if index-indx <= 0:
                return self[index+indx]
            else:
                return self[index-indx]
        
        # seed the one-hot vector
        one_hot_target = [0 for _ in range(len(self.__target_indicies))]
        # and set it
        one_hot_target[self.__target_indicies.index(target)] = 1

        return torch.tensor(data).long(), torch.tensor(data_found_mask).bool(), torch.tensor(one_hot_target).float()

    def __getitem__(self, index):
        # index the data
        data = self.data.iloc[index].copy()
        target = self.targets.iloc[index].copy()

        return self.__process(data, target, index)

    @functools.cache
    def val(self):
        """Return the validation set"""

        # collect dataset
        dataset = []

        print("Processing validation data...")

        # get it
        for index in tqdm(range(len(self.val_data))):
            try:
                dataset.append(self.__process(self.val_data.iloc[index].copy(),
                                            self.val_targets.iloc[index].copy()))
            except ValueError:
                continue # all zero ignore

        # return parts
        inp, mask, out = zip(*dataset)

        return torch.stack(inp).long(), torch.stack(mask).bool(), torch.stack(out).float()

    def __len__(self):
        return len(self.data)


# loading data
class NACCFutureDataset(Dataset):

    def __init__(self, file_path, feature_path,
              # skipping 2 impaired because of labeling inconsistency
                 target_feature="NACCUDSD", target_indicies=[1,3,4],
                 fold=0):
        """The NeuralPsycology Dataset

        Arguments:

        file_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_feature] (str): the name of feature to serve as the target
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [fold] (int): the n-th fold to select
        """

        # initialize superclass
        super(NACCFutureDataset, self).__init__()

        # Read the raw dataset
        self.raw_data = pd.read_csv(file_path)

        # get the fature variables
        with open(feature_path, 'r') as df:
            lines = df.readlines()
            self.features = [i.strip() for i in lines]

        # skip elements whose target is not in the list
        self.raw_data = self.raw_data[self.raw_data[target_feature].isin(target_indicies)] 
        # Get a list of participants
        participants = self.raw_data["NACCID"]

        # Drop the parcticipants 
        self.raw_data = self.raw_data.drop(columns="NACCID")

        # Make it a multiindex by combining the experiment ID with the participant
        # so we can index by participant as first pass
        index_participant_correlated = list(zip(participants, pd.RangeIndex(0, len(self.raw_data))))
        index_multi = pd.MultiIndex.from_tuples(index_participant_correlated, names=["Participant ID", "Entry ID"])
        self.raw_data.index = index_multi

        # k fold
        participants = self.raw_data.index.get_level_values(0)

        kf = KFold(n_splits=10, random_state=7, shuffle=True)

        participant_set = pd.Series(list(set(participants)))
        splits = kf.split(participant_set)
        train_ids, test_ids = list(splits)[fold]

        train_participants = participant_set[train_ids]
        test_participants = participant_set[test_ids]

        raw_data = self.raw_data


        # cosntruct a "diagnosis-in-nyears" column
        age_date_data = raw_data[["NACCUDSD", "NACCAGE"]]

        max_age_plus_fifty = max(raw_data.NACCAGE)+50

        def find_data(grp):
            """find the age til dementia

            grp: DataFrameGroupBy 
            """

            # get dementia indicies
            mci_indicies = grp[(grp.NACCUDSD==3)]
            dementia_indicies = grp[(grp.NACCUDSD==4)]

            # store demented
            ultimate_diag_type = (3 if(len(mci_indicies)) else 1) if (len(dementia_indicies) == 0) else 4

            # if length is 0 "the person never had dementia"
            if ultimate_diag_type == 1:
                # make the dementia age max+50 i.e. they will get dementia a long time later
                initial_dementia_age = max_age_plus_fifty
            elif ultimate_diag_type == 3:
                # get the first MCI age 
                initial_dementia_age = mci_indicies.iloc[0].NACCAGE
            else:
                # get the first dEMENITA age 
                initial_dementia_age = dementia_indicies.iloc[0].NACCAGE

            # get the number of years to the age of dementia
            age_til_dementia = (initial_dementia_age - grp.NACCAGE).apply(lambda x:0 if x<0 else x)

            # will be demented will be repeted n times to fit length
            return age_til_dementia, [ ultimate_diag_type for _ in range(len(grp)) ]

        # weird data gymnastics to unpeel the two columns
        print("Loading timeseries groups...")
        age_til_dementia, ultimate_diag_type = zip(*(age_date_data.groupby(level=0,axis=0).progress_apply(find_data)))
        print("Done!")

        # create series for the outputs
        age_til_dementia_series = pd.concat(age_til_dementia)
        ultimate_diag_series = pd.Series([j for i in ultimate_diag_type for j in i])

        # align the index
        ultimate_diag_series.index = age_til_dementia_series.index

        # dump in result
        raw_data.loc[:,"age_til_dementia"] = age_til_dementia_series.sort_index(level=1)
        raw_data.loc[:, "ultimate_diag_type"] = ultimate_diag_series.sort_index(level=1)


        # shuffle
        raw_data_sample = raw_data.sample(frac=1)
        age_til_dementia_sample = raw_data_sample["age_til_dementia"]
        raw_data_sample = raw_data_sample[((age_til_dementia_sample>bound[0])&(age_til_dementia_sample<bound[1]))|(age_til_dementia_sample > 50)] # > 50 is control 

        # shuffle again and sample based on median size
        median_count = raw_data_sample.ultimate_diag_type.value_counts().median()
        raw_data_sample = raw_data_sample.groupby("ultimate_diag_type").sample(n=int(median_count),
                                                                            replace=True)

        # shuffle again and sample based on median size
        raw_data_sample =  raw_data_sample.sample(frac=1)

        # Calculate the target data. this is not trivial.
        self.targets = raw_data_sample["ultimate_diag_type"]

        # then isolate the daata
        self.data = raw_data_sample[self.features] 

        # store the traget indicies
        self.__target_indicies = target_indicies

        # get number of features, by hoisting the get function up and getting length
        self._num_features = len(self.features)

        # get the actual intersections remaining
        current_participants = self.data.index.get_level_values(0)
        test_participants = current_participants.intersection(test_participants)
        train_participants = current_participants.intersection(train_participants)

        # crop the data for validatino
        self.val_data = self.data.loc[test_participants]
        self.val_targets = self.targets.loc[test_participants]

        self.data = self.data.loc[train_participants]
        self.targets = self.targets.loc[train_participants]

    def __process(self, data, target, index=None):
        # the discussed dataprep
        # if a data entry is <0 or >80, it is "not found"
        # so, we encode those values as 0 in the FEATURE
        # column, and encode another feature of "not-found"ness
        data_found = (data > 80) | (data < 0)
        data[data_found] = 0
        # then, the found-ness becomes a mask
        data_found_mask = (data_found)

        # if it is a sample with no tangible data
        # well give up and get another sample:
        if sum(~data_found_mask) == 0:
            if not index:
                raise ValueError("All-Zero found in validation!")
            indx = random.randint(2,5)
            if index-indx <= 0:
                return self[index+indx]
            else:
                return self[index-indx]
        
        # seed the one-hot vector
        one_hot_target = [0 for _ in range(len(self.__target_indicies))]
        # and set it
        one_hot_target[self.__target_indicies.index(target)] = 1

        return torch.tensor(data).long(), torch.tensor(data_found_mask).bool(), torch.tensor(one_hot_target).float()

    def __getitem__(self, index):
        # index the data
        data = self.data.iloc[index].copy()
        target = self.targets.iloc[index].copy()

        return self.__process(data, target, index)

    @functools.cache
    def val(self):
        """Return the validation set"""

        # collect dataset
        dataset = []

        print("Processing validation data...")

        # get it
        for index in tqdm(range(len(self.val_data))):
            try:
                dataset.append(self.__process(self.val_data.iloc[index].copy(),
                                            self.val_targets.iloc[index].copy()))
            except ValueError:
                continue # all zero ignore

        # return parts
        inp, mask, out = zip(*dataset)

        return torch.stack(inp).long(), torch.stack(mask).bool(), torch.stack(out).float()

    def __len__(self):
        return len(self.data)


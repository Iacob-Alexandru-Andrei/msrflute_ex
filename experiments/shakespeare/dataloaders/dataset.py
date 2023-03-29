# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import numpy as np
from core.dataset import BaseDataset
import torch
from experiments.shakespeare.dataloaders.preprocessing import SHAKESPEARE

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


class Dataset(BaseDataset):
    def __init__(
        self,
        data,
        transform=lambda p: torch.tensor(p),
        target_transform=lambda p: torch.tensor(p),
        test_only=False,
        user_idx=0,
        **kwargs
    ):

        self.transform = transform
        self.target_transform = target_transform
        self.characters = LEAF_CHARACTERS
        self.num_letters = len(self.characters)  # 80
        self.test_only = test_only
        self.user_idx = user_idx
        args = kwargs["args"]
        if data is None:
            (
                self.user_list,
                self.user_data,
                self.user_data_label,
                self.num_samples,
            ) = (
                [],
                {},
                {},
                [],
            )
        else:
            # Get all data
            (
                self.user_list,
                self.user_data,
                self.user_data_label,
                self.num_samples,
            ) = self.load_data(
                data=data,
                test_only=(test_only or user_idx == -1),
                filter_less=args["filter_less"],
                filter_more=args["filter_more"],
                data_loc=args["data_loc"],
            )

            if user_idx == -1:
                self.user = "test_only"
                self.data = self.user_data.values() if self.user_data else []
                self.labels = (
                    self.user_data_label.values() if self.user_data_label else []
                )

            else:
                if self.test_only:  # combine all data into single array
                    self.user = "test_only"
                    self.data = self.user_data.values() if self.user_data else []
                    self.labels = (
                        self.user_data_label.values() if self.user_data_label else []
                    )

                else:  # get a single user's data
                    if user_idx is None:
                        raise ValueError("in train mode, user_idx must be specified")

                    self.user = self.user_list[user_idx]
                    self.data = self.user_data[self.user]
                    self.labels = self.user_data_label[self.user]

    def word_to_indices(self, word):
        """Converts a sequence of characters into position indices in the
        reference string `self.characters`.
        Args:
            word (str): Sequence of characters to be converted.
        Returns:
            List[int]: List with positions.
        """
        indices = [self.characters.find(c) for c in word]
        return indices

    def __getitem__(self, idx):
        x = self.data[idx]["x"]
        y = self.data[idx]["y"]

        sentence_indices = self.word_to_indices(x)
        if self.transform is not None:
            sentence_indices = self.transform(sentence_indices)

        next_word_index = self.characters.find(y)
        if self.target_transform is not None:
            next_word_index = self.target_transform(next_word_index)

        return sentence_indices, next_word_index

    def __len__(self):
        return len(self.data)

    def load_data(self, data, test_only, filter_less, filter_more, data_loc):
        """Wrapper method to read/instantiate the dataset"""
        dataset = data
        if isinstance(data, str):
            dataset = SHAKESPEARE(
                data,
                filter_less=filter_less,
                filter_more=filter_more,
                data_loc=data_loc,
            )
            dataset = dataset.data
            print("Number of clients left:", len(dataset["users"]))

        users = dataset["users"]
        data = dataset["user_data"]
        labels = dataset["user_data_label"]
        num_samples = dataset["num_samples"]

        return users, data, labels, num_samples

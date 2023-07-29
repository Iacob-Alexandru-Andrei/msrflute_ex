# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np

from core.dataloader import BaseDataLoader
from experiments.reddit.dataloaders.dataset import Dataset
from torch.utils.data import RandomSampler
import sys
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_tokenizer


def get_collate(tokenizer):
    def collate(examples):
        nonlocal tokenizer
        if tokenizer._pad_token is None:
            return {"x": pad_sequence(examples, batch_first=True)}
        return {
            "x": pad_sequence(
                examples, batch_first=True, padding_value=tokenizer.pad_token_id
            )
        }

    return collate


class DataLoader(BaseDataLoader):
    def __init__(self, mode, num_workers=0, **kwargs):

        args = kwargs["args"]
        tokenizer = get_tokenizer(args["model"])
        self.batch_size = args["batch_size"]

        # FIXME fix for efficient sampling
        dataset = Dataset(
            data=kwargs["data"],
            test_only=(not mode == "train"),
            user_idx=kwargs.get("user_idx", None),
            args=args,
            tokenizer=tokenizer,
        )
        if mode == "train":

            sampler = RandomSampler(
                dataset,
                replacement=True,
                num_samples=self.batch_size * args["desired_max_samples"],
            )

            super().__init__(
                dataset,
                sampler=sampler,
                batch_size=self.batch_size,
                # shuffle=(mode == "train"),
                pin_memory=True,
                drop_last=(mode == "train"),
                num_workers=num_workers,
                timeout=60 if num_workers != 0 else 0,
                collate_fn=get_collate(tokenizer),
            )
        else:
            super().__init__(
                dataset,
                batch_size=self.batch_size,
                shuffle=(mode == "train"),
                pin_memory=True,
                drop_last=(mode == "train"),
                num_workers=num_workers,
                timeout=60 if num_workers != 0 else 0,
                collate_fn=get_collate(tokenizer),
            )

    # def collate_fn(self, batch):
    #     x, y = list(zip(*batch))
    #     # x, y = np.array(x), np.array(y)
    #     batched_x = torch.stack(x)
    #     batched_y = torch.tensor(y)
    #     return {"x": batched_x, "y": batched_y}

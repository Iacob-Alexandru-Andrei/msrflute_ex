# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np

from core.dataloader import BaseDataLoader
from experiments.openImg.dataloaders.dataset import Dataset
from torch.utils.data import RandomSampler


class DataLoader(BaseDataLoader):
    def __init__(self, mode, num_workers=0, **kwargs):
        args = kwargs["args"]
        self.batch_size = args["batch_size"]

        # FIXME fix for efficient sampling
        dataset = Dataset(
            data=kwargs["data"],
            test_only=(not mode == "train"),
            user_idx=kwargs.get("user_idx", None),
            args=args,
        )
        if mode == "train":
            super().__init__(
                dataset,
                batch_size=self.batch_size,
                shuffle=(mode == "train"),
                pin_memory=True,
                drop_last=(mode == "train"),
                num_workers=num_workers,
                timeout=60 if num_workers != 0 else 0,
                collate_fn=self.collate_fn,
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
                collate_fn=self.collate_fn,
            )

    def collate_fn(self, batch):
        x, y = list(zip(*batch))
        # x, y = np.array(x), np.array(y)
        batched_x = torch.stack(x)
        batched_y = torch.tensor(y)
        return {"x": batched_x, "y": batched_y}

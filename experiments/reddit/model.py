import importlib.util
import math
import sys

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from transformers import (
    AdamW,
    AlbertTokenizer,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    MobileBertForPreTraining,
)
import os
from core.model import BaseModel
from transformers import AlbertTokenizer
import types

sys.path.append("/home/aai30/nfs-share/projects")

from FedScale.fedscale.dataloaders.nlp import mask_tokens


class ALBERT(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_config["model_setup_config"],
        )

        self.model_config = model_config

        self.tokenizer = AlbertTokenizer.from_pretrained(
            model_config["model_str"], do_lower_case=True
        )
        self.net = AutoModelWithLMHead.from_config(config)

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        """Performs forward step and computes the loss"""

        class BUNCH:
            def __init__(self, **kwds):
                self.__dict__.update(kwds)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok_config = BUNCH(**self.model_config)
        data = input["x"]

        data, target = mask_tokens(
            data, tokenizer=self.tokenizer, args=tok_config, device=device
        )
        data, target = data.to(device), target.to(device)

        outputs = self.net(data, labels=target)
        loss = outputs[0]

        return loss

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features, labels = input["x"].to(device), input["y"].to(device)
        output = self.net.forward(features)

        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()

        return {"output": output, "acc": accuracy, "batch_size": n_samples}

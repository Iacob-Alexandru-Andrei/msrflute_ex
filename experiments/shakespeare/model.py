import torch
import torch.nn as nn
from torch.nn import functional as F


from core.model import BaseModel

import sys

sys.path.append("/home/aai30/nfs-share/projects")


""" 
    The ResNet models are taken from FedML repository. For more information regarding this model, 
    please refer to https://github.com/FedML-AI/FedML/blob/master/python/fedml/model/cv/resnet_gn.py.
"""


LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


class ShakespeareLeafNet(nn.Module):  # type: ignore
    """Create Shakespeare model for LEAF baselines.
    Args:
        chars (str, optional): String of possible characters (letters+digits).
            Defaults to LEAF_CHARACTERS.
        seq_len (int, optional): Length of each sequence. Defaults to 80.
        hidden_size (int, optional): Size of hidden layer. Defaults to 256.
        embedding_dim (int, optional): Dimension of embedding. Defaults to 8.
    """

    def __init__(
        self,
        chars=LEAF_CHARACTERS,
        seq_len=80,
        hidden_size=256,
        embedding_dim=8,
    ):
        super().__init__()
        self.dict_size = len(chars)
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(self.dict_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,  # Notice batch is first dim now
        )
        self.decoder = nn.Linear(self.hidden_size, self.dict_size)

    def forward(self, sentence):
        """Forwards sentence to obtain next character.
        Args:
            sentence (torch.Tensor): Tensor containing indices of characters
        Returns:
            torch.Tensor: Vector encoding position of predicted character
        """
        encoded_seq = self.encoder(sentence)  # (batch, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(encoded_seq)  # (batch, seq_len, hidden_size)
        pred = self.decoder(h_n[-1])
        return pred


class FluteShakespeare(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = ShakespeareLeafNet(chars=LEAF_CHARACTERS)

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        """Performs forward step and computes the loss"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features, labels = input["x"].to(device), input["y"].to(device)
        output = self.net.forward(features)

        return F.cross_entropy(output, labels.long())

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features, labels = input["x"].to(device), input["y"].to(device)
        output = self.net.forward(features)

        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()

        return {"output": output, "acc": accuracy, "batch_size": n_samples}

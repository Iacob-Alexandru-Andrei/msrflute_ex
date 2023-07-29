from core.model import BaseModel
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import shufflenet_v2_x2_0


class SHUFFLENET(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = shufflenet_v2_x2_0()

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

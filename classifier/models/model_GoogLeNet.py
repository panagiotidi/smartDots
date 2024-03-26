import torch
from torch.nn import Linear, Module, Softmax, Sigmoid, ReLU
from torchvision.models import GoogLeNet_Weights

from config import total_classes, regression


class GoogLeNet(Module):

    def __init__(self):
        super().__init__()

        # GoogLeNet model
        self.model_googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights=GoogLeNet_Weights.DEFAULT)

        if regression == 'continuous':
            self.linear = Linear(in_features=1000, out_features=1)
        else:
            self.linear = Linear(in_features=1000, out_features=total_classes)

        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

    def forward(self, input):
        out = self.model_googlenet(input)
        out = self.relu(out)

        # Train mode returns GoogLeNetOutputs class with logits
        if not torch.is_tensor(out):
            out = out.logits

        out = self.linear(out)

        if regression == 'continuous':
            predictions = out.squeeze(-1)
        elif regression == 'categorical':
            predictions = self.softmax(out)
        else:
            predictions = self.sigmoid(out)

        return predictions
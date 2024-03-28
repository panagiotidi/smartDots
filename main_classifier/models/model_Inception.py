import torch
import torchvision
from torch.nn import Module, Softmax, Sigmoid, ReLU
from config import total_classes, regression


class Inception(Module):

    def __init__(self):
        super().__init__()

        # Inception3 model
        if regression == 'continuous':
            self.inception = torchvision.models.Inception3(num_classes=1, aux_logits=True, init_weights=True)
        else:
            self.inception = torchvision.models.Inception3(num_classes=total_classes, aux_logits=True, init_weights=True)

        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

    def forward(self, input):
        out = self.inception(input)

        # Train mode returns InceptionOutputs class with logits
        if not torch.is_tensor(out):
            out = out.logits

        # out = self.relu(out)
        if regression == 'continuous':
            predictions = out.squeeze(-1)
        elif regression == 'categorical':
            predictions = self.softmax(out)
        else:
            predictions = self.sigmoid(out)

        return predictions
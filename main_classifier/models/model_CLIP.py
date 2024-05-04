import clip
import torch
from torch.nn import Linear, Module, Softmax, Sigmoid, ReLU, Tanh
from torchvision.transforms.v2 import Normalize
from transformers import CLIPModel, CLIPImageProcessor, CLIPFeatureExtractor, CLIPProcessor
from config import total_classes, regression, device


class Clip(Module):

    def __init__(self):
        super().__init__()

        # Clip model
        # model_version = 'openai/clip-vit-base-patch32'
        model_preprocess_version = 'ViT-L/14@336px'
        self.clip_model, self.preprocess = clip.load(model_preprocess_version, device, jit=False)
        self.clip_model.float()
        self.clip_model.train(False)

        self.linear0 = Linear(in_features=768, out_features=500)
        self.linear1 = Linear(in_features=500, out_features=500)
        # self.linear2 = Linear(in_features=500, out_features=1000)

        if regression == 'continuous':
            self.linear = Linear(in_features=500, out_features=1)
        else:
            self.linear = Linear(in_features=500, out_features=total_classes)

        # self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.relu = ReLU()
        self.tanh = Tanh()

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module_name, module in self.named_children():
            if module_name != 'clip_model':
                module.train(mode)
        self.clip_model.train(False)
        return self

    def eval(self):
        return self.train(False)

    def get_preprocess(self):
        return self.preprocess

    def forward(self, input):
        features = self.clip_model.encode_image(input)
        out = self.linear0(features)
        # out = self.relu(out)
        # out = self.linear1(out)
        # out = self.tanh(out)
        out = self.linear(out)

        if regression == 'continuous':
            predictions = out.squeeze(-1)
        elif regression == 'categorical_abs' or regression == 'categorical_prob':
            # predictions = self.softmax(out)
            predictions = out
        else:
            predictions = self.sigmoid(out)

        return predictions
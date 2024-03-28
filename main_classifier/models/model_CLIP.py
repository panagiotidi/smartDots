import clip
import torch
from torch.nn import Linear, Module, Softmax, Sigmoid, ReLU
from torchvision.transforms.v2 import Normalize
from transformers import CLIPModel, CLIPImageProcessor, CLIPFeatureExtractor, CLIPProcessor
from config import total_classes, regression, device


class Clip(Module):

    def __init__(self):
        super().__init__()

        # Clip model
        # model_version = 'openai/clip-vit-base-patch32'
        model_preprocess_version = 'ViT-L/14@336px'
        # r:CLIPModel
        # r.bfloat16()
        # model_version = 'ViT-B/32'
        # model_version = 'openai/clip-vit-large-patch14-336'
        # self.preprocess: CLIPModel = CLIPImageProcessor.from_pretrained(model_version)
        self.clip_model, self.preprocess = clip.load(model_preprocess_version, device, jit=False)
        self.clip_model.float()
        # self.clip_model.bfloat16()
        self.clip_model.train(False)
        # p: CLIPModel
        # print(self.clip_model.get_parameter('positional_embedding'))
        # self.clip_model.positional_embedding
        # self.clip_model, _ = clip.load(model_preprocess_version, device=device, jit=False)
        # self.processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # self.preprocess = CLIPImageProcessor.from_pretrained(model_version)
        # self.clip_model = CLIPFeatureExtractor.from_pretrained(model_version)
        self.linear0 = Linear(in_features=768, out_features=1000)

        if regression == 'continuous':
            self.linear = Linear(in_features=1000, out_features=1)
        else:
            self.linear = Linear(in_features=1000, out_features=total_classes)

        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module_name, module in self.named_children():
            if module_name != 'clip_model':
                module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def get_preprocess(self):
        return self.preprocess

    def forward(self, input):
        # input = self.normalize(input)
        features = self.clip_model.encode_image(input)
        out = self.linear0(features)
        out = self.relu(out)
        out = self.linear(out)

        if regression == 'continuous':
            predictions = out.squeeze(-1)
        elif regression == 'categorical':
            predictions = self.softmax(out)
        else:
            predictions = self.sigmoid(out)

        return predictions
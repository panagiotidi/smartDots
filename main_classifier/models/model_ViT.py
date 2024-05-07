import torch
from torch.nn import Linear, Module, Softmax, Sigmoid, ReLU
from transformers import AutoImageProcessor, ViTForImageClassification
from config import total_classes, regression, device, ViT_model


class ViT(Module):

    def __init__(self):
        super().__init__()

        # ViT model
        self.image_processor = AutoImageProcessor.from_pretrained(ViT_model)
        self.vit_model = ViTForImageClassification.from_pretrained(ViT_model,
                                                                   num_labels=total_classes,
                                                                   ignore_mismatched_sizes=True)

        if regression == 'continuous':
            self.linear = Linear(in_features=1000, out_features=1)
        else:
            self.linear = Linear(in_features=1000, out_features=total_classes)

        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

    def forward(self, input):

        inputs = self.image_processor(input, return_tensors="pt", do_rescale=False).to(device)
        out = self.vit_model(**inputs)

        # Train mode returns GoogLeNetOutputs class with logits
        if not torch.is_tensor(out):
            out = out.logits

        if regression == 'continuous':
            predictions = out.squeeze(-1)
        elif regression == 'categorical_abs' or regression == 'categorical_prob':
            # predictions = self.softmax(out)
            predictions = out
        else:
            predictions = self.sigmoid(out)

        return predictions

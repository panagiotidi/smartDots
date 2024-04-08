import torchvision
from torch.nn import Linear, Module, Softmax, Dropout, ReLU, Sigmoid
from torchvision.models import ResNet, ResNet101_Weights
from config import total_classes, regression


class ResNet(Module):

    def __init__(self):
        super().__init__()

        # ResNet model
        self.model_resnet: ResNet = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        # self.model_resnet: ResNet = torchvision.models.resnet50(weights=None)

        if regression == 'continuous':
            self.linear = Linear(in_features=1000, out_features=1)
        else:
            self.linear = Linear(in_features=1000, out_features=total_classes)

        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.dropout = Dropout(0.4)
        self.relu = ReLU()

    def forward(self, input):
        out = self.model_resnet(input)
        # print(out.shape)
        out = self.linear(out)
        out = self.relu(out)
        # out = self.dropout(out)

        # build the entire model
        # x = out.output
        # print(input.shape)
        # x = AvgPool2d((3, 2), stride=(2, 1))(input)
        # print(x.shape)
        # x = Linear(in_features=x.shape[3], out_features=512).to(device)(x)
        # x = ReLU()(x)
        # x = Dropout(0.2)(x)
        # print(x.shape)
        # x = Linear(in_features=x.shape[3], out_features=256).to(device)(x)
        # x = ReLU()(x)
        # x = Dropout(0.2)(x)
        # x = Linear(in_features=x.shape[3], out_features=128).to(device)(x)
        # x = ReLU()(x)
        # x = Dropout(0.2)(x)
        # x = Linear(in_features=x.shape[3], out_features=64).to(device)(x)
        # x = ReLU()(x)
        # out = Dropout(0.2)(x)
        # print(out.shape)

        if regression == 'continuous':
            predictions = out.squeeze(-1)
        elif regression == 'categorical_abs':
            predictions = self.softmax(out)
        else:
            predictions = self.sigmoid(out)

        return predictions



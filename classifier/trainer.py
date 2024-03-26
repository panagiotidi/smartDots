import os
import sys

import torch.nn as nn
from numpy import nan
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import torch
from torcheval.metrics import MulticlassConfusionMatrix
from torchvision.transforms import ToTensor, Compose, Resize

from classifier.dataloader.FishLoader import FishDataset
from classifier.models.model_GoogLeNet import GoogLeNet
from classifier.models.model_net import Net
from classifier.models.model_resnet import ResNet
from classifier.models.model_Inception import Inception


from config import BATCH_SIZE, epochs, clean_data_path, subsample_fraction, device, learning_rate, weights, \
    filter_species, model_name, regression, total_classes, weight_decay


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


if __name__ == '__main__':

    #################### Define class weights ###############################3

    class_weights = weights

    #################### Define preprocess ###############################3

    # trainTransforms = transforms.Compose([resize, hFlip, vFlip, rotate, transforms.ToTensor()])
    # valTransforms = transforms.Compose([resize, transforms.ToTensor()])

    # initialize our data augmentation functions
    # resize = transforms.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH))
    # hFlip = transforms.RandomHorizontalFlip(p=0.25)
    # vFlip = transforms.RandomVerticalFlip(p=0.25)
    # rotate = transforms.RandomRotation(degrees=15)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms = Compose([
                Resize(512),
                # transforms.CenterCrop(224),
                ToTensor(),
                # Grayscale(3),
                # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    #################### Data preparation ###############################3

    trainDataset = FishDataset(os.path.join(clean_data_path, 'train'), preprocess=transforms, fraction=subsample_fraction, filter_species=filter_species)
    valDataset = FishDataset(os.path.join(clean_data_path, 'val'), preprocess=transforms, fraction=subsample_fraction, filter_species=filter_species)

    # # create training and validation set dataloaders
    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    print('Train DataLoader length:', len(trainDataLoader))
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE)
    print('Val DataLoader length:', len(valDataLoader))

    #################### Model ###############################3

    model = str_to_class(model_name)().to('mps')
    print('Model: ', model.__class__)
    #################### Loss ###############################3
    print('Regression type: ', regression)

    if regression == 'continuous':
        criterion = nn.MSELoss().to(device)
    else:
        # criterion = nn.MSELoss().to(device)
        # criterion = R2Score().to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device), reduction='mean')

    print('Loss function: ', criterion)

    metric_conf_matrix = MulticlassConfusionMatrix(num_classes=total_classes)

    #################### Optimizer, Scheduler ###############################3

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = LinearLR(optimizer, total_iters=epochs)
    #################### Train process ###############################3

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        all_labels = []
        all_preds = []
        for i, data in enumerate(trainDataLoader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            # print('inputs.shape:', inputs.shape)
            # print('outputs.shape:', outputs)
            # print('labels.shape:', labels)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # metric.update(outputs, labels)
            # print('R2:',  metric.compute())

            running_loss += loss.item()
            print('batch loss:', loss.item())

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainDataLoader)}")

        val_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():

            for i, data in enumerate(valDataLoader, 0):
                inputs, labels = data
                logps = model(inputs)
                # print(logps.shape)
                labels.to(device)
                # print(labels.shape)
                loss = criterion(logps, labels)
                # print('val batch loss:', loss.item())
                val_loss += loss.item()
                metric_conf_matrix.update(logps.argmax(1).clone().detach().cpu(), labels.argmax(1).clone().detach().cpu())
                all_labels = all_labels + list(labels.argmax(1).clone().detach().cpu().numpy())
                all_preds = all_preds + list(logps.argmax(1).clone().detach().cpu())

            # ps = torch.exp(logps)
            # top_p, top_class = ps.topk(1, dim=1)
            # equals = top_class == labels.view(*top_class.shape)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss / len(valDataLoader)}")
        print('Confusion matrix\n', metric_conf_matrix.compute())
        print(classification_report(all_labels, all_preds, zero_division=nan))

        # # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # train_losses.append(running_loss / len(trainDataLoader))
        # val_losses.append(val_loss / len(valDataLoader))
        # print(f"Epoch {epoch + 1}/{config.epochs}.. "
        #   # f"Train loss: {running_loss / print_every:.3f}.. "
        #   f"Val loss: {val_loss / len(valDataLoader):.3f}.. "
        #   f"Test accuracy: {accuracy / len(valDataLoader):.3f}")

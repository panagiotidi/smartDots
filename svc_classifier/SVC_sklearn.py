import os

import clip
import torch
import torchvision
from numpy import nan

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet101_Weights, ResNet, GoogLeNet_Weights, Inception3
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from transformers import ViTForImageClassification, AutoImageProcessor

from config import clean_data_path, subsample_fraction, filter_species, BATCH_SIZE, regression, model_name, device, \
    total_classes, C, metric_max_diff, ViT_model, ViT_model_proc, GoogLeNet_model, ViT_clip_model_preprocess_version
from dataloader.FishLoader import FishDataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from utils import compute_max_diff

print('Model: ', model_name)
if model_name == 'Clip':
    # Load the model
    model, preprocess = clip.load(ViT_clip_model_preprocess_version, device)
else:
    if model_name == 'ResNet':
        model: ResNet = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
    elif model_name == 'GoogLeNet':
        model = torch.hub.load('pytorch/vision:v0.10.0', GoogLeNet_model, weights=GoogLeNet_Weights.DEFAULT)
    elif model_name == 'Inception':
        model: Inception3 = torchvision.models.Inception3(num_classes=500, aux_logits=True, init_weights=True)
    elif model_name == 'ViT':
        image_processor = AutoImageProcessor.from_pretrained(ViT_model_proc)
        model = ViTForImageClassification.from_pretrained(ViT_model, ignore_mismatched_sizes=True)
    else:
        exit('No known model:', model_name)
    if model_name == 'Inception':
        preprocess = Compose([
            # Resize(512),
            ToTensor()
        ])
    else:
        preprocess = Compose([
            ToTensor(),
        ])

    model.to(device)

trainDataset = FishDataset(os.path.join(clean_data_path, 'train'), preprocess=preprocess, fraction=subsample_fraction, filter_species=filter_species)
valDataset = FishDataset(os.path.join(clean_data_path, 'val'), preprocess=preprocess, fraction=subsample_fraction, filter_species=filter_species)

# create training and validation set dataloaders
# trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
# print('Train DataLoader length:', len(trainDataLoader))
# valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE)
# print('Val DataLoader length:', len(valDataLoader))


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=BATCH_SIZE)):

            if model_name == 'Clip':
                features = model.encode_image(images.to(device))
            elif model_name == 'ViT':
                inputs = image_processor(images, return_tensors="pt", do_rescale=False).to(device)
                features = model(**inputs).logits
            else:
                if model_name == 'Inception':
                    features = model(images).logits
                else:
                    features = model(images)
            all_features.append(features)

            if regression == 'continuous':
                all_labels.append(labels, dim=1)
            elif regression == 'categorical_prob' or regression == 'ordinal':
                exit('sklearn classification with probabilities/arrays as targets not allowed.')
            else: #regression == 'categorical_abs':
                all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Calculate the image features
train_features, train_labels = get_features(trainDataset)
test_features, test_labels = get_features(valDataset)

print('train_labels', train_labels)
# -------------------------------------------------------------------------------

classifier = SVC(gamma='auto', C=C, class_weight='balanced')
classifier.fit(train_features, train_labels)
print('Fitting finished!')

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
print('Predicted test sample.')

print('Confusion matrix\n', confusion_matrix(predictions, test_labels, labels=range(0, total_classes)))

accuracy = np.mean((test_labels == predictions).astype(float)) * 100
print(f"Accuracy = {accuracy:.3f}")

print(classification_report(test_labels, predictions, zero_division=nan))

print('Metric:', compute_max_diff(test_labels, predictions, metric_max_diff))
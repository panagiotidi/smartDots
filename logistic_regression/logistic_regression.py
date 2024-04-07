import os

import clip
import torch
from numpy import nan

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassConfusionMatrix
from tqdm import tqdm
from config import clean_data_path, subsample_fraction, filter_species, BATCH_SIZE, total_classes, regression, device, C
from main_classifier.dataloader.FishLoader import FishDataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report


model_version = 'ViT-L/14@336px'
# model_version = 'google/vit-hugepatch14–224-in21k'
# model_version = 'ViT-B/32'

# Load the model
model, preprocess = clip.load(model_version, device)

trainDataset = FishDataset(os.path.join(clean_data_path, 'train'), preprocess=preprocess, fraction=subsample_fraction, filter_species=filter_species)
valDataset = FishDataset(os.path.join(clean_data_path, 'val'), preprocess=preprocess, fraction=subsample_fraction, filter_species=filter_species)

# create training and validation set dataloaders
trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
print('Train DataLoader length:', len(trainDataLoader))
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE)
print('Val DataLoader length:', len(valDataLoader))


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=BATCH_SIZE)):

            features = model.encode_image(images.to('mps'))
            all_features.append(features)

            if regression == 'continuous':
                all_labels.append(labels, dim=1)
            elif regression == 'categorical':
                all_labels.append(torch.argmax(labels, dim=1))
            else:
                all_labels.extend([labels])

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Calculate the image features
train_features, train_labels = get_features(trainDataset)
test_features, test_labels = get_features(valDataset)

print('train_labels', train_labels)

# -------------------------------------------------------------------------------

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=C, max_iter=100, verbose=1, class_weight='balanced')
classifier.fit(train_features, train_labels)
print('Fitting finished!')

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
print('Predicted test sample.')
accuracy = np.mean((test_labels == predictions).astype(float)) * 100
print(f"Accuracy = {accuracy:.3f}")

print(classification_report(test_labels, predictions, zero_division=nan))


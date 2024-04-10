import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from config import device, total_classes, regression
from utils import unify_label, getOldestAge, inverse_weights


def get_label_from_row(row):
    age = row['ModalAge_AllReaders']
    if regression == 'continuous':
        return float(age)
    elif regression == 'categorical_abs':
        try:
            label = total_classes * [0.0]
            label[int(age)] = 1.0
            return label
        except:
            print('Error! Switching to unified label!!')
            return unify_label(row, total_classes)
    elif regression == 'categorical_prob':
        return unify_label(row, total_classes)
    # ordenal
    # See https://stackoverflow.com/questions/38375401/neural-network-ordinal-classification-for-age
    else:
        try:
            label = total_classes * [0.0]
            for i in range(int(age) + 1):
                label[i] = 1.0
            return label
        except:
            exit('I have not implemented ordenal regression for age represented by non integers/probabilities yet!\n'
                 'The column representing the age needs to be an exact integer if you pick ordenal regression!')


class FishDataset(Dataset):

    def __init__(self, pics_path, preprocess, fraction=1.0, filter_species=None):
        self.pics_path = pics_path
        self.preprocess = preprocess

        dataset = pd.read_csv(os.path.join(pics_path, "data.csv"))
        print('Dataset size:', len(dataset))

        print('Unique species count:', dataset['Species'].value_counts())

        if filter_species:
            dataset = dataset[dataset['Species'] == filter_species]
            print('After keeping only ' + filter_species + ' species:', len(dataset))

        if regression == 'categorical_prob':
            dataset["OldestAge"] = dataset.apply(getOldestAge, axis=1)
            dataset = dataset[dataset['OldestAge'] <= total_classes - 1]
        else:
            dataset = dataset[dataset['ModalAge_AllReaders'] <= total_classes - 1]
        print('After removing rare ages data size:', len(dataset))

        dataset = dataset.sample(frac=fraction, replace=False, random_state=1)
        print('Subsampled dataset size:', len(dataset))
        dataset.reset_index(drop=True)

        print('Unique ages count:', dataset['ModalAge_AllReaders'].value_counts())
        # sorted_dict = dict(sorted(my_dict.items()))
        self.class_weights = list(dict(sorted(dataset['ModalAge_AllReaders'].value_counts().to_dict().items())).values())

        self.data = []
        self.labels = []
        for index, row in dataset.iterrows():
            img_base_name = row['Species'] + '_' + str(row['ModalAge_AllReaders']) + '_' + row['ImageName']
            img_path = os.path.join(pics_path, img_base_name)
            self.data.append(img_path)
            label = get_label_from_row(row)
            self.labels.append(label)

        self.dataset = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]

        # print('pic path:', img_path)
        input_image = Image.open(img_path)
        # print('before:', list(input_image.getdata()))
        input_image = self.preprocess(input_image)
        # print('after:', input_image)

        classes_annotations = self.labels[idx]
        # print(classes_annotations)
        classes_annotations = torch.as_tensor(classes_annotations).float().to(device)

        return input_image.float().to(device), classes_annotations

    def getClassWeights(self):
        return torch.Tensor(inverse_weights(self.class_weights))
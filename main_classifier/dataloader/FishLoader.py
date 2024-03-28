import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from config import device, total_classes, regression
from utils import unify_label


def get_label_from_row(row):
    age = row['ModalAge_AllReaders']
    if regression == 'continuous':
        return float(age)
    elif regression == 'categorical':
        try:
            label = total_classes * [0.0]
            label[int(age)] = 1.0
            return label
        except:
            print('Error! Switching to unified label!!')
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

        if filter_species:
            dataset = dataset[dataset['Species'] == filter_species]
            print('After keeping only ' + filter_species + ' species:', len(dataset))

        dataset = dataset[dataset['ModalAge_AllReaders'] <= total_classes - 1]
        print('After removing rare ages data size:', len(dataset))

        dataset = dataset.sample(frac=fraction, replace=False, random_state=1)
        print('Subsampled dataset size:', len(dataset))
        dataset.reset_index(drop=True)

        print('Unique ages count:', dataset['ModalAge_AllReaders'].value_counts())

        self.data = []
        self.labels = []
        for index, row in dataset.iterrows():
            img_base_name = row['Species'] + '_' + str(row['ModalAge_AllReaders']) + '_' + row['ImageName']
            img_path = os.path.join(pics_path, img_base_name)
            self.data.append(img_path)
            label = get_label_from_row(row)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]

        input_image = Image.open(img_path)
        input_image = self.preprocess(input_image)

        classes_annotations = self.labels[idx]
        classes_annotations = torch.as_tensor(classes_annotations).float().to(device)

        return input_image.float().to(device), classes_annotations

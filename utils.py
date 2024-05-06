import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from numpy import nan


smartdots_url = 'http://smartdots.ices.dk/sampleimages/'


def create_name(row):
    # img_name: str = row['URL'].lower()
    img_name: str = row['URL']
    # new_img_name = img_name.strip().replace(smartdots_url, '').replace('/', '_')
    new_img_name = img_name.strip().replace(smartdots_url, '').split('/')[-1]
    return new_img_name


def compute_max_diff(target, preds, metric_max_diff):
    diff = np.absolute(preds - target)
    return len(np.where(diff <= metric_max_diff)[0]) / len(target)


def is_label_ok(row):
    row_label: str = row['ModalAge_AllReaders']
    age = int(row_label)
    if age < 0:
        return False
    return True


def unify_label(row, num_classes):
    label = num_classes * [0.0]
    row_label: str = row['ReadersAges']
    # print('row:', row_label)
    splits = row_label.replace(' ', '').split('-')
    for split in splits:
        sp = split.split(':')
        class_number = sp[1]
        if int(class_number) >= num_classes:
            print('Error! ', row['ModalAge_AllReaders'], row['ReadersAges'])
            return None
        observ_amount = sp[0]
        label[int(class_number)] = int(observ_amount)
    return probs_from_lst(label)


def getOldestAge(row):
    row_label: str = row['ReadersAges']
    splits = row_label.replace(' ', '').split('-')
    max_age = 0
    for split in splits:
        sp = split.split(':')
        class_number = sp[1]
        if int(class_number) > max_age:
            max_age = int(class_number)
    return max_age


def inverse_weights(lst):
    sum = 0.0
    for l in lst:
        sum = sum + 1 / l
    return [1 / (l * sum) for l in lst]


def probs_from_lst(lst):
    return [l / sum(lst) for l in lst]


def print_separate_confusions(dataset, predictions, abs_labels):
    dataset['abs_labels'] = abs_labels
    dataset['predictions'] = predictions
    unique_species = dataset.Species.unique()
    print('unique_species', unique_species)
    gb = dataset.groupby(['Species'])
    for species in unique_species:

        table = gb.get_group((species,) )
        preds = table['predictions']
        ground = table['abs_labels']
        print('Classification report for', species, ':\n', classification_report(ground, preds, zero_division=nan))
        print('Confusion for', species, ':\n', confusion_matrix(ground, preds))
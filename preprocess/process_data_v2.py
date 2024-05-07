import os
import traceback
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from config import VAL_SPLIT, original_path, clean_data_path, data_csv, total_classes, INPUT_HEIGHT, INPUT_WIDTH, \
    MinMaxAgeDif, MaxModalAge
from preprocess.preprocess_object_periklis import periklis_preprocess
from preprocess.preprocess_object_sofia2 import pre_process_sofia2
from utils import create_name, is_label_ok
import shutil
import glob

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def is_local(row: pd.core.series.Series):
    split_pics = glob.glob(os.path.join(original_path, row['Species'], row['Code'] + '*'))

    if len(split_pics) > 0:
        for split_pic_path in split_pics:
            split_picture = row.copy(deep=True)
            split_picture['Code'] = split_pic_path.split('/')[-1]
            rows_list.append(split_picture)
        return True
    else:
        return False


def process_and_save(dataset, set_path):
    out_path = os.path.join(clean_data_path, set_path)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    i = 0
    for index, row in dataset.iterrows():
        try:

            img_path = os.path.join(original_path, row['Species'], row['Code'])
            img = cv2.imread(img_path)

            img = pre_process_sofia2(img)
            img = cv2.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))

            # Save .jpg image
            img_base_name = row['Species'] + '_' + str(row['ModalAge_AllReaders']) + '_' + row['Code']
            if os.path.exists(out_path + img_base_name):
                print('File exists! Replacing:', row['Code'])

            write_res = cv2.imwrite(out_path + img_base_name, img)
            if not write_res:
                print('Write failed:', img_base_name)
                dataset = dataset[dataset['Code'] != row['Code']]
            if not os.path.exists(out_path + img_base_name):
                print('File not written for some unknown reason!', img_base_name)
            i = i + 1

        except:
            # If error in saving, remove from dataset
            print("Resize or write failed:", img_base_name)
            dataset = dataset[dataset['Code'] != row['Code']]
            traceback.print_exc()

    print('Saved total pics:', i)
    dataset.reset_index(drop=True)
    dataset.to_csv(out_path + 'data.csv')


if __name__ == '__main__':
    # Read the CSV file
    data = pd.read_csv(data_csv)
    # data = data[0:5000]
    print(data.shape)

    # Remove url duplicates
    data = data[data['URL'].duplicated(False) == False]
    data.reset_index(drop=True)
    print('Data shape after removing url duplicates:', data.shape)

    data['Code'] = data.apply(create_name, axis=1)
    # Remove Code duplicates
    data = data[data['Code'].duplicated(False) == False]
    print('Data shape after removing Code duplicates:', data.shape)

    data = data[data['MinMaxAgeDif'] <= MinMaxAgeDif]
    print('Data shape after removing MinMaxAgeDif more than ' + str(MinMaxAgeDif) + ':', data.shape)

    # Remove images with more than total_classes modal age
    data = data[data['ModalAge_AllReaders'] <= MaxModalAge]
    print('Data shape after discarding modal age more than 20 years:', data.shape)

    # Remove bad labels
    data['label_ok'] = data.apply(is_label_ok, axis=1)
    data = data[data['label_ok'] == True]
    data.reset_index(drop=True)
    print('Data shape after removing bad labels:', data.shape)

    rows_list = []
    # Remove non-existing locally saved
    data['is_local'] = data.apply(is_local, axis=1)
    data = data[data['is_local'] == True]
    # data.reset_index(drop=True)
    print('Data shape after removing non existing images locally:', data.shape)

    # df_all represents the dataframe with all the otoliths split in _0.jpg, _1.jpg, ...
    df_all = pd.DataFrame(rows_list)
    print('df_all:', df_all.shape)
    # df_all.to_csv('tmp.csv')

    # Data split in train and test
    print(df_all['ModalAge_AllReaders'].value_counts())
    train, val = train_test_split(df_all, stratify=df_all['ModalAge_AllReaders'], test_size=VAL_SPLIT)
    print('Data shape final:', train.shape, val.shape)

    print('Species stats:', df_all['Species'].value_counts())
    print('ModalAge_AllReaders stats:', df_all['ModalAge_AllReaders'].value_counts())

    # Data resize, save in folders and csv too
    sets = [(train, 'train/'), (val, 'val/')]
    for (dataset, set_path) in sets:
        process_and_save(dataset, set_path)

import os
import traceback
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from config import VAL_SPLIT, original_path, clean_data_path, data_csv, total_classes
from preprocess.preprocess_object_periklis import periklis_preprocess
from utils import create_name, is_label_ok
from process_object import pre_process
import shutil

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def is_local(row):
    img_base_name = row['Species'] + '_' + str(row['ModalAge_AllReaders']) + '_' + row['ImageName']
    img_path = os.path.join(original_path, img_base_name)
    if os.path.exists(img_path):
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

            img_base_name = row['Species'] + '_' + str(row['ModalAge_AllReaders']) + '_' + row['ImageName']
            img_path = os.path.join(original_path, img_base_name)
            img = cv2.imread(img_path)
            # img = cv2.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, dstCn=3)
            # img = cv2.merge([img,img,img])

            img = pre_process(img)
            # img = periklis_preprocess(img)

            # Save .jpg image
            if os.path.exists(out_path + img_base_name):
                print('File exists! Replacing:', row['ImageName'])

            write_res = cv2.imwrite(out_path + img_base_name, img)
            if not write_res:
                print('Write failed:', img_base_name)
                dataset = dataset[dataset['ImageName'] != row['ImageName']]
            if not os.path.exists(out_path + img_base_name):
                print('File not written for some unknown reason!', img_base_name)
            i = i + 1

        except:
            # If error in saving, remove from dataset
            print("Resize or write failed:", img_base_name)
            dataset = dataset[dataset['ImageName'] != row['ImageName']]
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

    data['ImageName'] = data.apply(create_name, axis=1)
    # Remove ImageName duplicates
    # data = data[data['ImageName'].duplicated(False) == False]
    # data.reset_index(drop=True)
    # print('Data shape after removing ImageName duplicates:', data.shape)

    # Remove non-existing locally saved
    data['is_local'] = data.apply(is_local, axis=1)
    data = data[data['is_local'] == True]
    # data.reset_index(drop=True)
    print('Data shape after removing non existing images locally:', data.shape)

    # Remove bad labels
    data['label_ok'] = data.apply(is_label_ok, axis=1)
    data = data[data['label_ok'] == True]
    data.reset_index(drop=True)
    print('Data shape after removing bad labels:', data.shape)

    # Create uniform labels and remove those above a specific number (rare) total_classes
    # data['uni_label'] = data.apply(unify_label, axis=1)
    # print('Data shape after creating unified labels:', data.shape)
    # data = data.dropna()
    # data.reset_index(drop=True)
    # print('Data shape after removing rare labels (big ages):' , data.shape)

    # Create jpeg names for png files
    # data['new_name'] = data.apply(create_name, axis=1)
    # print('Data shape after creating new names for pngs:', data.shape)

    # Remove ImageName duplicates
    data = data[data['ImageName'].duplicated(False) == False]
    data.reset_index(drop=True)
    print('Data shape after removing ImageName duplicates:', data.shape)

    # Remove images with more than total_classes modal age
    data = data[data['ModalAge_AllReaders'] <= 20]
    print('Data shape after discarding modal age more than 20 years:' , data.shape)

    # Data split in train and test
    print(data['ModalAge_AllReaders'].value_counts())
    train, val = train_test_split(data, stratify=data['ModalAge_AllReaders'], test_size=VAL_SPLIT)
    print('Data shape final:' , train.shape, val.shape)

    print(data['Species'].value_counts())
    print(data['ModalAge_AllReaders'][data['Species'] == 'Pollachius virens'].value_counts())

    # Data resize, save in folders and csv too
    sets = [(train, 'train/'), (val, 'val/')]
    for (dataset, set_path) in sets:
        process_and_save(dataset, set_path)

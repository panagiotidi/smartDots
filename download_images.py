import urllib.request
import pandas as pd
from config import original_path, metadata_csv
from utils import create_name


def download_and_save(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        print("Failed to download:", url)


if __name__ == '__main__':
    # Read the CSV file
    data = pd.read_csv(metadata_csv)
    # data = data[0:50]
    print(data.columns)
    print('Data shape:', data.shape)

    # Remove url duplicates
    data = data[data['URL'].duplicated(False) == False]
    data.reset_index(drop=True)
    print('Data shape after removing url duplicates:', data.shape)

    data['ImageName'] = data.apply(create_name, axis=1)
    data = data[data['ImageName'].duplicated(False) == False]
    data.reset_index(drop=True)
    print('Data shape after removing ImageName duplicates:', data.shape)

    for index, row in data.iterrows():
        url = row['URL']
        img_base_name = row['Species'] + '_' + str(row['ModalAge_AllReaders']) + '_' + row['ImageName']
        download_and_save(url, original_path + img_base_name)


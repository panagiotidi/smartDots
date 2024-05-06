import os
import shutil
import pandas as pd
from urllib3 import PoolManager
from config import original_path, metadata_csv, MinMaxAgeDif
from utils import create_name

http = PoolManager()


def download_and_save(dataset, url, filename):
    try:
        with open(filename, 'wb') as out:
            r = http.request('GET', url, preload_content=False)
            if r.status == 200:
                shutil.copyfileobj(r, out)
            else:
                raise Exception("404, image not found:")
    except:
        dataset = dataset[dataset['URL'] != url]
        print("Failed to download:", url, 'Removing it from the database..')
    return dataset


if __name__ == '__main__':
    # Read the CSV file
    data = pd.read_csv(metadata_csv)
    # data = data[0:50]
    print('Columns:', data.columns)
    print('Data shape:', data.shape)

    # Remove url duplicates
    data = data[data['URL'].duplicated(False) == False]
    data.reset_index(drop=True)
    print('Data shape after removing url duplicates:', data.shape)

    data['ImageName'] = data.apply(create_name, axis=1)
    data = data[data['ImageName'].duplicated(False) == False]
    data.reset_index(drop=True)
    print('Data shape after removing ImageName duplicates:', data.shape)

    final_data = data.copy(deep=True)

    for index, row in data.iterrows():
        url = row['URL']
        img_base_name = row['Species'] + '_' + str(row['ModalAge_AllReaders']) + '_' + row['ImageName']
        final_data = download_and_save(final_data, url, os.path.join(original_path, img_base_name))

    final_data.to_csv('new_data.csv')

import os

import cv2

from config import original_path, clean_data_path
from preprocess.preprocess_object_periklis import periklis_preprocess
from preprocess.process_object_sofia import pre_process

for filename in os.listdir(original_path):
    f = os.path.join(original_path, filename)
    if os.path.isfile(f) and f.endswith('jpg'):
        img = cv2.imread(f)
        print(f)
        img = pre_process(img)
        # img = periklis_preprocess(img)
        cv2.imwrite(os.path.join(clean_data_path, filename), img)
        print(f)
from utils import inverse_weights

metadata_csv = '/Users/sofia/PycharmProjects/smartDots/data/original_all_accordance/smartDots_Species_v2.csv'

original_path = 'data/original_all_accordance'
# clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/all_accordance_processed512x512'
# clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/all_accordance_processed512x512_2'
clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/all_accordance_processed256x256'

# set the input height and width
INPUT_HEIGHT = 224 #(416, 416)
INPUT_WIDTH = 224

# Options: 'ResNet', 'GoogLeNet', 'ResNetUNet', 'Inception'
model_name = 'Inception'

# Options:
#  'categorical': classification, logistic regression
#  'continuous' : continuous, non-linear
#  'ordinal' : Ordinal regression is half-way between classification and real-valued regression. When you perform multiclass classification of your ordinal data, you are assigning the same penalty whenever your classifier predicts a wrong class, no matter which one.
regression = 'categorical'

total_classes = 6

# weights = total_classes * [1.0]
# weights = inverse_weights([69, 171, 141, 189, 199, 173, 118, 102, 68, 31, 32])
# weights = inverse_weights([283,668,589,804,791,690,504,339,247,156,108])
weights = inverse_weights([16, 39, 98, 271, 414, 444])



# Possible values: 'Pleuronectes platessa', 'Ammodytes', 'Solea solea', 'Pollachius pollachius', 'Micromesistius poutassou', 'Pollachius virens', 'Gadus morhua', etc.
# filter_species = None
filter_species = 'Pleuronectes platessa'

device = 'mps'
subsample_fraction = 1.0
BATCH_SIZE = 12
VAL_SPLIT = 0.2
epochs = 4
learning_rate = 1e-04



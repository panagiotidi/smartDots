from utils import inverse_weights

# ---------------------------------------------------------------------------------------------------------------
# Settings for downloading data
metadata_csv = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024/SmartDotsSummaryExtraction-2.csv'
original_path = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024'
# original_path = '/Users/sofia/PycharmProjects/smartDots/subset'
MinMaxAgeDif = 3
# ---------------------------------------------------------------------------------------------------------------
# Settings for preprocessing data
data_csv = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024/data.csv'
clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024_processed_simple'
# clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/subset/out_sofia'
# clean_data_path = '/Users/sofia/PycharmProjects/smartDots/data/all_accordance_processed256x256'

# ---------------------------------------------------------------------------------------------------------------

# set the input height and width
INPUT_HEIGHT = 512  # (416, 416)
INPUT_WIDTH = 512

# Options: 'ResNet', 'GoogLeNet', 'ResNetUNet', 'Inception', 'Clip', 'ViT'
model_name = 'ViT'

# Options:
#  'categorical_abs': classification, logistic regression, one absolute age
#  'categorical_prob': classification, logistic regression, age probability range
#  'continuous' : continuous, non-linear
#  'ordinal' : Ordinal regression is half-way between classification and real-valued regression. When you perform multiclass classification of your ordinal data, you are assigning the same penalty whenever your classifier predicts a wrong class, no matter which one.
regression = 'categorical_prob'

total_classes = 6

# weights = total_classes * [1.0]
# weights = inverse_weights([69, 171, 141, 189, 199, 173, 118, 102, 68, 31, 32])
# weights = inverse_weights([283,668,589,804,791,690,504,339,247,156,108])
weights = inverse_weights([15, 47, 103, 261, 429, 430])

# Possible values: None, 'Pleuronectes platessa', 'Ammodytes', 'Solea solea', 'Pollachius pollachius', 'Micromesistius poutassou', 'Pollachius virens', 'Gadus morhua', etc.
filter_species = 'Pleuronectes platessa'

device = 'mps'
subsample_fraction = 1.0
BATCH_SIZE = 12
VAL_SPLIT = 0.2

# ----------- For trainer -------------#
epochs = 2
learning_rate = 1e-04
weight_decay = 0.01
# ----------- For SVC_regression -------------#
C = 0.6
# ----------- Metric settings -------------#
metric_max_diff = 1